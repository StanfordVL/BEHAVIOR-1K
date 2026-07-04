import builtins
import sys
from types import SimpleNamespace

from omnigibson.utils.lazy_import_utils import LazyImporter


class _NewtonLazyImporter(LazyImporter):
    def __init__(self):
        super().__init__("", None)
        self.carb = SimpleNamespace(
            input=SimpleNamespace(
                KeyboardInput=_KeyboardConstants(),
                KeyboardEventType=SimpleNamespace(
                    KEY_PRESS="KEY_PRESS",
                    KEY_REPEAT="KEY_REPEAT",
                    KEY_RELEASE="KEY_RELEASE",
                ),
            )
        )


class _KeyboardConstants:
    def __getattr__(self, name):
        return name


if getattr(builtins, "OMNIGIBSON_NEWTON_NATIVE", False):
    sys.modules[__name__] = _NewtonLazyImporter()
else:
    sys.modules[__name__] = LazyImporter("", None)
