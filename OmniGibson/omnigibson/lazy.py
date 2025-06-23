import sys

from omnigibson.utils.lazy_import_utils import LazyImporter

sys.modules[__name__] = LazyImporter("", None)
