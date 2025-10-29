"""
Quick training script for 1-2 epochs on custom CHASIS dataset
Simplified version for testing and visualization
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# Add surfemb to path
sys.path.insert(0, str(Path(__file__).parent))

from surfemb.workspace_dino.model_forward_c2f_dino_unet import Gen_corr
from surfemb.workspace_dino.dino_feat_model import DINO_feat
from surfemb.data.instance_tudl import BopInstanceDataset
from surfemb.data import config as data_config
from surfemb.data.obj import load_objs
from surfemb.data.std_auxs import RgbLoader

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


def main():
    print("="*60)
    print("Quick Training - RAG-6DPose (1-2 epochs)")
    print("="*60)

    # Configuration
    DATASET_ROOT = Path("/media/sujith/Project/NOARK_CV/RAG-6DPose-code/chasis_dataset")
    CAD_FEATURES_PATH = Path("/media/sujith/Project/NOARK_CV/RAG-6DPose-code/cad_features")
    MODELS_PATH = Path("/media/sujith/Project/NOARK_CV/RAG-6DPose-code/models")
    OUTPUT_DIR = Path("./quick_train_output")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Training params
    BATCH_SIZE = 4  # Small batch for quick testing
    NUM_EPOCHS = 2
    GPUS = [0]
    RES_CROP = 224
    N_OBJS = 1  # Only obj_000001

    print(f"\nConfiguration:")
    print(f"  Dataset: {DATASET_ROOT}")
    print(f"  CAD features: {CAD_FEATURES_PATH}")
    print(f"  Models: {MODELS_PATH}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Crop resolution: {RES_CROP}")
    print(f"  Number of objects: {N_OBJS}")

    # Check files
    print(f"\n" + "="*60)
    print("Checking required files...")
    print("="*60)

    cad_feat_file = CAD_FEATURES_PATH / "obj_000001_dino_feat.pt"
    models_info = MODELS_PATH / "models_info.json"
    model_ply = MODELS_PATH / "obj_000001.ply"

    if not cad_feat_file.exists():
        print(f"ERROR: CAD features not found: {cad_feat_file}")
        return
    if not models_info.exists():
        print(f"ERROR: models_info.json not found: {models_info}")
        return
    if not model_ply.exists():
        print(f"ERROR: PLY model not found: {model_ply}")
        return

    print(f"✓ CAD features: {cad_feat_file}")
    print(f"✓ models_info: {models_info}")
    print(f"✓ PLY model: {model_ply}")

    # Configuration
    ds_config = data_config.config['chasis']
    obj_ids = [1]  # obj_000001

    # Load objects first (needed for model initialization and auxs)
    print(f"\n" + "="*60)
    print("Loading object models...")
    print("="*60)

    try:
        objs, loaded_obj_ids = load_objs(MODELS_PATH, obj_ids=obj_ids, show_progressbar=False)
        print(f"✓ Loaded {len(objs)} object(s)")
        for obj in objs:
            print(f"  - obj_{obj.obj_id:06d}: scale={obj.scale:.2f}, offset={obj.offset}")

        # Model expects 3 objects, duplicate if we have fewer
        while len(objs) < 3:
            objs.append(objs[-1])  # Duplicate the last object
            print(f"  (Duplicated object {len(objs)} for compatibility)")
    except Exception as e:
        print(f"ERROR loading objects: {e}")
        import traceback
        traceback.print_exc()
        return

    # Initialize DINO model
    print(f"\n" + "="*60)
    print("Loading DINO feature extractor...")
    print("="*60)

    try:
        dino_model = DINO_feat()
        dino_model.load_model()
        for param in dino_model.parameters():
            param.requires_grad = False
        print(f"✓ DINO model loaded (frozen)")
    except Exception as e:
        print(f"ERROR loading DINO model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Initialize Gen_corr model
    print(f"\n" + "="*60)
    print("Initializing model...")
    print("="*60)

    try:
        model = Gen_corr(
            objs=objs,
            n_objs=N_OBJS,
            dino_model=dino_model,
            one_obj_idx=0,
        )
        print(f"✓ Model initialized")
        print(f"  Encoder: ResNet + UNet")
        print(f"  DINO feature dim: 768")
        print(f"  Query embedding dim: 256")
    except Exception as e:
        print(f"ERROR initializing model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Load training data with proper auxiliary pipeline
    print(f"\n" + "="*60)
    print("Setting up dataset with augmentation pipeline...")
    print("="*60)

    try:
        # Get the proper auxiliary pipeline from the model
        auxs = model.get_auxs(objs, RES_CROP)

        train_ds = BopInstanceDataset(
            dataset_root=DATASET_ROOT,
            pbr=False,
            synt=False,
            test=False,
            cfg=ds_config,
            obj_ids=obj_ids,
            min_visib_fract=0.1,
            min_px_count_visib=1024,
            auxs=auxs,
        )
        print(f"✓ Training dataset loaded: {len(train_ds)} samples")
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    # Custom collate function to handle negative strides in numpy arrays
    def custom_collate(batch):
        """Collate function that handles numpy arrays with negative strides"""
        def make_contiguous(obj):
            if isinstance(obj, np.ndarray) and not obj.flags['C_CONTIGUOUS']:
                return obj.copy()  # Make contiguous copy
            return obj

        # Process batch to fix negative strides
        batch = [{k: make_contiguous(v) for k, v in item.items()} for item in batch]

        # Use default collate
        from torch.utils.data.dataloader import default_collate
        return default_collate(batch)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Disable multiprocessing to simplify debugging
        pin_memory=True,
        collate_fn=custom_collate,
    )

    print(f"✓ DataLoader created: {len(train_loader)} batches")

    # Setup trainer
    print(f"\n" + "="*60)
    print("Setting up trainer...")
    print("="*60)

    checkpoint_callback = ModelCheckpoint(
        dirpath=OUTPUT_DIR,
        filename='chasis-epoch{epoch:02d}-step{step:06d}',
        save_top_k=2,
        monitor='step',
        mode='max',
        save_last=True,
    )

    # Configure trainer (gpus parameter removed in newer PyTorch Lightning)
    trainer_kwargs = {
        'max_epochs': NUM_EPOCHS,
        'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
        'callbacks': [checkpoint_callback],
        'log_every_n_steps': 1,
        'enable_checkpointing': True,
        'default_root_dir': OUTPUT_DIR,
    }

    # Add devices parameter if using GPU
    if torch.cuda.is_available():
        trainer_kwargs['devices'] = GPUS

    trainer = pl.Trainer(**trainer_kwargs)

    print(f"✓ Trainer configured")
    print(f"  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Max epochs: {NUM_EPOCHS}")

    # Train
    print(f"\n" + "="*60)
    print("Starting training...")
    print("="*60)

    try:
        trainer.fit(model, train_loader)
        print(f"\n" + "="*60)
        print("Training completed successfully!")
        print("="*60)
        print(f"\nCheckpoints saved to: {OUTPUT_DIR}")

        # List saved checkpoints
        ckpt_files = list(OUTPUT_DIR.glob("*.ckpt"))
        if ckpt_files:
            print(f"\nSaved checkpoints:")
            for ckpt in ckpt_files:
                size_mb = ckpt.stat().st_size / 1024 / 1024
                print(f"  - {ckpt.name} ({size_mb:.1f} MB)")

    except Exception as e:
        print(f"\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
