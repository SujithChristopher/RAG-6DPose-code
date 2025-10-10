from pathlib import Path
import argparse

import torch.utils.data
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

from .. import utils
from ..data import obj, instance_tudl
from ..data.config import config
from ..surface_embedding import SurfaceEmbeddingModel
from ..workspace_dino.model_forward_c2f_dino_unet import Gen_corr

def worker_init_fn(*_):
    # each worker should only use one os thread
    # numpy/cv2 takes advantage of multithreading by default
    import os
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    import cv2
    cv2.setNumThreads(0)

    # random seed
    import numpy as np
    np.random.seed(None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    # parser.add_argument('model_path')
    parser.add_argument('--n-valid', type=int, default=400)
    parser.add_argument('--res-data', type=int, default=256)
    parser.add_argument('--res-crop', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32) #32
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--min-visib-fract', type=float, default=0.1)
    parser.add_argument('--max-steps', type=int, default=500_000)
    parser.add_argument('--gpus',  default=[0,1])
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ckpt', default=None)
    parser.add_argument('--no-synth', dest='synth', action='store_false')
    parser.add_argument('--real', action='store_true')

    args = parser.parse_args()
    debug = args.debug
    root = Path('/home/bop/datasets') / args.dataset
    cfg = config[args.dataset]

    # load objs
    objs, obj_ids = obj.load_objs(root / cfg.model_folder)
    assert len(obj_ids) > 0
    print(obj_ids)
    # model
    if args.ckpt:
        assert args.dataset == Path(args.ckpt).name.split('-')[0]
        model = SurfaceEmbeddingModel.load_from_checkpoint(args.ckpt)
    else:
        from ..workspace_dino.dino_feat_model import DINO_feat
        dino_model = DINO_feat()
        dino_model.load_model()
        for param in dino_model.parameters():
            param.requires_grad = False
        model = Gen_corr(objs, n_objs=len(obj_ids),dino_model=dino_model, one_obj_idx=0)
    # datasets
    auxs = model.get_auxs(objs, args.res_crop)
    data = utils.EmptyDataset()
    if args.synth:
        data += instance_tudl.BopInstanceDataset(
            dataset_root=root, pbr=True, synt=False, test=False, cfg=cfg, obj_ids=obj_ids, auxs=auxs,
            min_visib_fract=args.min_visib_fract, scene_ids=[1] if debug else None,infer_idx=10
        )
        data += instance_tudl.BopInstanceDataset(
            dataset_root=root, pbr=False, synt=True, test=False, cfg=cfg, obj_ids=obj_ids, auxs=auxs,
            min_visib_fract=args.min_visib_fract, scene_ids=[1] if debug else None,infer_idx=10
        )
    if args.real:
        data_real = instance_tudl.BopInstanceDataset(
            dataset_root=root, pbr=False,synt=False, test=False, cfg=cfg, obj_ids=obj_ids, auxs=auxs,
            min_visib_fract=args.min_visib_fract, scene_ids=[1] if debug else None,infer_idx=10
        )
        if args.synth:
            data = data + data_real
        else:
            data = data_real
    

    n_valid = args.n_valid
    data_train, data_valid = torch.utils.data.random_split(
        data, (len(data) - n_valid, n_valid),
        generator=torch.Generator().manual_seed(0),
    )

    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=torch.get_num_threads() if args.num_workers is None else args.num_workers,
        persistent_workers=True, shuffle=True,
        worker_init_fn=worker_init_fn, pin_memory=True,
    )
    
    loader_args_valid = dict(
        batch_size=args.batch_size,
        num_workers=torch.get_num_threads() if args.num_workers is None else args.num_workers,
        persistent_workers=True, shuffle=False,
        worker_init_fn=worker_init_fn, pin_memory=True,
    )
    loader_train = torch.utils.data.DataLoader(data_train, drop_last=True, **loader_args)
    loader_valid = torch.utils.data.DataLoader(data_valid, **loader_args_valid)
    # train
    log_dir = Path('data/logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    import os
    os.environ["WANDB_API_KEY"] = "your key"
    # os.environ["WANDB_MODE"]="offline"
    run = wandb.init(project='proj', dir=log_dir)
    run.name = run.id

    logger = pl.loggers.WandbLogger(experiment=run)
    logger.log_hyperparams(args)

    model_ckpt_cb = pl.callbacks.ModelCheckpoint(dirpath='data/models/', save_top_k=0, save_last=True)
    model_ckpt_cb.CHECKPOINT_NAME_LAST = f'{args.dataset}-{run.id}'
    trainer = pl.Trainer(
        max_epochs = 24,
        accelerator='gpu', devices=args.gpus,
        strategy='ddp_find_unused_parameters_true',  
        # resume_from_checkpoint=args.ckpt,
        logger=logger, max_steps=args.max_steps,
        callbacks=[
            pl.callbacks.LearningRateMonitor(),
            model_ckpt_cb,
        ],
        val_check_interval=min(1., n_valid / len(data) * 50)  # spend ~1/50th of the time on validation
    )
    trainer.fit(model, loader_train, loader_valid)


if __name__ == '__main__':
    main()
