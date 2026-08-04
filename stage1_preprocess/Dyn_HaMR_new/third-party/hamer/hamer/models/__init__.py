import os
from .mano_wrapper import MANO
from .hamer import HAMER
from .discriminator import Discriminator

from ..utils.download import cache_url
# from ..configs import CACHE_DIR_HAMER

def download_models(folder=None):
    """Download checkpoints and files for running inference.
    """
    
    os.makedirs(folder, exist_ok=True)
    download_files = {
        "hamer_demo_data.tar.gz"      : ["https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz", folder],
    }
    
    for file_name, url in download_files.items():
        output_path = os.path.join(url[1], file_name)
        if not os.path.exists(os.path.join(folder, '_DATA', 'hamer_ckpts')):
            print("Downloading file: " + file_name, "to ", output_path)
            # output = gdown.cached_download(url[0], output_path, fuzzy=True)
            output = cache_url(url[0], output_path)
            assert os.path.exists(output_path), f"{output} does not exist"

            # if ends with tar.gz, tar -xzf
            if file_name.endswith(".tar.gz"):
                print("Extracting file: " + file_name, 'to ', folder)
                os.system("tar -xvf " + output_path + f" -C {folder}")

def load_hamer(root=None):
    from pathlib import Path
    from ..configs import get_config
    checkpoint_path = os.path.join(root, '_DATA/hamer_ckpts/checkpoints/hamer.ckpt')
    model_cfg = str(Path(checkpoint_path).parent.parent / 'model_config.yaml')
    model_cfg = get_config(model_cfg, update_cachedir=True)

    # Defrost the config to allow modifications
    model_cfg.defrost()

    # Update MANO configuration
    model_cfg.MANO.DATA_DIR = os.path.join(root, '_DATA/data/')
    model_cfg.MANO.MEAN_PARAMS = os.path.join(root, '_DATA/data/mano_mean_params.npz')
    model_cfg.MANO.MODEL_PATH = os.path.join(root, '_DATA/data/mano')
    model_cfg.EXTRA.FOCAL_LENGTH = 500.0
    # model_cfg.MANO.MEAN_PARAMS = os.path.join(root, '_DATA/hamer_ckpts/checkpoints/hamer.ckpt')

    # Override some config values, to crop bbox correctly
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        assert model_cfg.MODEL.IMAGE_SIZE == 256, f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]

    # Update config to be compatible with demo
    if 'PRETRAINED_WEIGHTS' in model_cfg.MODEL.BACKBONE:
        model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')

    # Freeze the config after modifications
    model_cfg.freeze()

    model = HAMER.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
    return model, model_cfg
