from copy import deepcopy
from omegaconf import OmegaConf
from omnigibson.learning.utils.functional_utils import is_sequence, is_mapping, call_once
from omnigibson.learning.utils.print_utils import to_scientific_str


_NO_INSTANTIATE = "__no_instantiate__"  # return config as-is


@call_once(on_second_call="noop")
def register_omegaconf_resolvers():
    import numpy as np

    OmegaConf.register_new_resolver("scientific", lambda v, i=0: to_scientific_str(v, i))
    OmegaConf.register_new_resolver("_optional", lambda v: f"_{v}" if v else "")
    OmegaConf.register_new_resolver("optional_", lambda v: f"{v}_" if v else "")
    OmegaConf.register_new_resolver("_optional_", lambda v: f"_{v}_" if v else "")
    OmegaConf.register_new_resolver("__optional", lambda v: f"__{v}" if v else "")
    OmegaConf.register_new_resolver("optional__", lambda v: f"{v}__" if v else "")
    OmegaConf.register_new_resolver("__optional__", lambda v: f"__{v}__" if v else "")
    OmegaConf.register_new_resolver("iftrue", lambda cond, v_default: cond if cond else v_default)
    OmegaConf.register_new_resolver("ifelse", lambda cond, v1, v2="": v1 if cond else v2)
    OmegaConf.register_new_resolver("ifequal", lambda query, key, v1, v2: v1 if query == key else v2)
    OmegaConf.register_new_resolver("intbool", lambda cond: 1 if cond else 0)
    OmegaConf.register_new_resolver("mult", lambda *x: np.prod(x).tolist())
    OmegaConf.register_new_resolver("add", lambda *x: sum(x))
    OmegaConf.register_new_resolver("div", lambda x, y: x / y)
    OmegaConf.register_new_resolver("intdiv", lambda x, y: x // y)

    # try each key until the key exists. Useful for multiple classes that have different
    # names for the same key
    def _try_key(cfg, *keys):
        for k in keys:
            if k in cfg:
                return cfg[k]
        raise KeyError(f"no key in {keys} is valid")

    OmegaConf.register_new_resolver("trykey", _try_key)
    # replace `resnet.gn.ws` -> `resnet_gn_ws`, because omegaconf doesn't support
    # keys with dots. Useful for generating run name with dots
    OmegaConf.register_new_resolver("underscore_to_dots", lambda s: s.replace("_", "."))

    def _no_instantiate(cfg):
        cfg = deepcopy(cfg)
        cfg[_NO_INSTANTIATE] = True
        return cfg

    OmegaConf.register_new_resolver("no_instantiate", _no_instantiate)


def omegaconf_to_dict(cfg, resolve: bool = True, enum_to_str: bool = False):
    """
    Convert arbitrary nested omegaconf objects to primitive containers

    WARNING: cannot use tree lib because it gets confused on DictConfig and ListConfig
    """
    kw = dict(resolve=resolve, enum_to_str=enum_to_str)
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, **kw)
    elif is_sequence(cfg):
        return type(cfg)(omegaconf_to_dict(c, **kw) for c in cfg)
    elif is_mapping(cfg):
        return {k: omegaconf_to_dict(c, **kw) for k, c in cfg.items()}
    else:
        return cfg


def omegaconf_save(cfg, *paths: str, resolve: bool = True):
    """
    Save omegaconf to yaml
    """
    from .file_utils import f_join

    OmegaConf.save(cfg, f_join(*paths), resolve=resolve)
