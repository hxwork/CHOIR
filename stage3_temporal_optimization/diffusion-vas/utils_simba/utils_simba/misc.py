import os
from omegaconf import OmegaConf
from packaging import version
import json


# # ============ Register OmegaConf Recolvers ============= #
# OmegaConf.register_new_resolver('calc_exp_lr_decay_rate', lambda factor, n: factor**(1./n))
# OmegaConf.register_new_resolver('add', lambda a, b: a + b)
# OmegaConf.register_new_resolver('sub', lambda a, b: a - b)
# OmegaConf.register_new_resolver('mul', lambda a, b: a * b)
# OmegaConf.register_new_resolver('div', lambda a, b: a / b)
# OmegaConf.register_new_resolver('idiv', lambda a, b: a // b)
# OmegaConf.register_new_resolver('basename', lambda p: os.path.basename(p))
# # ======================================================= #


def prompt(question):
    inp = input(f"{question} (y/n)").lower().strip()
    if inp and inp == 'y':
        return True
    if inp and inp == 'n':
        return False
    return prompt(question)


def load_config(*yaml_files, cli_args=[]):
    yaml_confs = [OmegaConf.load(f) for f in yaml_files]
    cli_conf = OmegaConf.from_cli(cli_args)
    conf = OmegaConf.merge(*yaml_confs, cli_conf)
    OmegaConf.resolve(conf)
    return conf


def config_to_primitive(config, resolve=True):
    return OmegaConf.to_container(config, resolve=resolve)


def dump_config(path, config):
    with open(path, 'w') as fp:
        OmegaConf.save(config=config, f=fp)

def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def parse_version(ver):
    return version.parse(ver)

def merge_json(source_json_f1,source_json_f2, output_json_f):
    with open(source_json_f1, 'r') as f:
        source1 = json.load(f)
    with open(source_json_f2, 'r') as f:
        source2 = json.load(f)
    source1.update(source2)
    with open(output_json_f, 'w') as f:
        json.dump(source1, f, indent=4) 


def list_folders(directory):
    try:
        # List all entries in the directory
        entries = os.listdir(directory)
        # Filter out entries that are directories
        folders = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
        return folders
    except FileNotFoundError:
        return []
    except PermissionError:
        return []