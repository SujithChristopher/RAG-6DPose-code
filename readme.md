Thanks to Surfemb! Our code is based on https://github.com/rasmushaugaard/surfemb.

For Dataset please ref to: BOP https://bop.felk.cvut.cz/datasets/. 

The input of our method is RGB images.

For Evaluation, please ref to: BOP toolkit https://github.com/thodan/bop_toolkit

python scripts/eval_bop19_pose.py --renderer_type=vispy --result_filenames=NAME_OF_CSV_WITH_RESULTS

For training:

python -m surfemb.scripts.train_matching_dino_unet_tudl tudl --real

Please modify the wandb key of os.environ["WANDB_API_KEY"]

For custom dataset(images, models), you can use our preprocess codes in custom_data_proc dir to get BOP format.

