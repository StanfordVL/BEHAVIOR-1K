import sys
from typing import TYPE_CHECKING, Any

from omnigibson.utils.lazy_import_utils import LazyImporter

sys.modules[__name__] = LazyImporter("", None)

if TYPE_CHECKING:
    # This module's namespace is replaced at runtime (see above) by a LazyImporter
    # that resolves arbitrary submodules/attributes on the fly. Pyright can't see
    # through that swap, so tell it about the dynamic attribute access via PEP 562.
    def __getattr__(name: str) -> Any: ...
