import sys
from omnigibson.utils.lazy_import_utils import LazyImporter

_import_structure = {
  "omni.isaac.core.utils.prims": ["get_prim_at_path"],
}

sys.modules[__name__] = LazyImporter(__name__, globals()["__file__"], _import_structure)