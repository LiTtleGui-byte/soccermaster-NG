import torch
from runtime_option import runtime_option
from utils.misc import set_seed
from configs.util import load_super_config, update_config, yaml_to_dict
import os
from accelerate import Accelerator
from accelerate.state import PartialState
from accelerate.utils import DistributedDataParallelKwargs
from data.build import build_dataset

def train_engine(config: dict):
    # Init some settings:
    assert "EXP_NAME" in config and config["EXP_NAME"] is not None, "Please set the experiment name."
    outputs_dir = config["OUTPUTS_DIR"] if config["OUTPUTS_DIR"] is not None \
        else os.path.join("./outputs/", config["EXP_NAME"])

    # Init Accelerator at beginning:
    # accelerator = Accelerator()
    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False)]
    )
    state = PartialState()
    # Also, we set the seed:
    set_seed(config["SEED"])
    
    # from issue: https://github.com/pytorch/pytorch/issues/11201
    # Set the sharing strategy (to avoid error: too many open files):
    torch.multiprocessing.set_sharing_strategy('file_system')   # if not, raise error: too many open files.
    # torch.autograd.set_detect_anomaly(True)
    
    # Init Logger:
    # TODO: finish logger code
    
    train_dataset = build_dataset(config=config)

if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # from issue: https://github.com/pytorch/pytorch/issues/11201
    # import torch.multiprocessing
    # torch.multiprocessing.set_sharing_strategy('file_system')

    # Get runtime option:
    opt = runtime_option()
    cfg = yaml_to_dict(opt.config_path)

    # Loading super config:
    if opt.super_config_path is not None:   # the runtime option is priority
        cfg = load_super_config(cfg, opt.super_config_path)
    else:                                   # if not, use the default super config path in the config file
        cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])

    # Combine the config and runtime into config dict:
    cfg = update_config(config=cfg, option=opt)

    # Call the "train_engine" function:
    # train_engine(config=cfg)