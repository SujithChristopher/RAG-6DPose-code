from collections import defaultdict


class DatasetConfig:
    model_folder = 'models'
    train_folder = 'train'
    test_folder = 'test'
    img_folder = 'rgb'
    depth_folder = 'depth'
    img_ext = 'png'
    depth_ext = 'png'


config = defaultdict(lambda *_: DatasetConfig())

config['tless'] = tless = DatasetConfig()
tless.model_folder = 'models_cad'
tless.test_folder = 'test_primesense'
tless.train_folder = 'train_primesense'

config['hb'] = hb = DatasetConfig()
hb.test_folder = 'test_primesense'

config['itodd'] = itodd = DatasetConfig()
itodd.depth_ext = 'tif'
itodd.img_folder = 'gray'
itodd.img_ext = 'tif'

# CHASIS dataset configuration
config['chasis'] = chasis = DatasetConfig()
chasis.model_folder = 'models'
chasis.train_folder = 'train_real'
chasis.test_folder = 'test'
chasis.img_folder = 'rgb'
chasis.depth_folder = 'depth'
chasis.img_ext = 'png'
chasis.depth_ext = 'png'
