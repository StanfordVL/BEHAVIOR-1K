from bddl.backend_abc import BDDLBackend

_AVAILABLE_BACKENDS = ["iGibson"]
_backend = None


def set_backend(backend_name):
    global _backend
    if backend_name == "iGibson":
        from igibson.tasks.bddl_backend import IGibsonBDDLBackend
        _backend = IGibsonBDDLBackend()
    else:
        raise ValueError("Invalid backend. Currently supported backends: %s." % ", ".join(_AVAILABLE_BACKENDS))

    if not isinstance(_backend, BDDLBackend):
        raise ValueError("Backends must implement bddl.backend_abc.BDDLBackend.")



def get_backend():
    if _backend is None:
        raise ValueError(
            "Before calling bddl functions, a backend must be set using bddl.set_backend(backend_name). "
            "Available backend names: %s." % ", ".join(_AVAILABLE_BACKENDS))

    return _backend