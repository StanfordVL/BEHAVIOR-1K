import importlib
from types import ModuleType


class LazyImporter(ModuleType):
    """Replace a module's global namespace with me to support lazy imports of submodules and members."""

    def __init__(self, module_name, module):
        super().__init__("lazy_" + module_name)
        self._module_path = module_name
        self._module = module
        self._not_module = set()
        self._submodules = {}

    def __getattr__(self, name: str):
        # First, try the argument as a module name.
        if name not in self._not_module:
            submodule = self._get_module(name)
            if submodule:
                return submodule
            else:
                # Record module not found so that we don't keep looking.
                self._not_module.add(name)

        # If it's not a module name, try it as a member of this module.
        try:
            return getattr(self._module, name)
        except:
            raise AttributeError(f"module {self.__name__} has no attribute {name}") from None

    def _get_module(self, module_name: str):
        """Recursively create and return a LazyImporter for the given module name."""

        # Get the fully qualified module name by prepending self._module_path
        if self._module_path:
            module_name = f"{self._module_path}.{module_name}"

        if module_name in self._submodules:
            return self._submodules[module_name]

        try:
            wrapper = LazyImporter(module_name, importlib.import_module(module_name))
            self._submodules[module_name] = wrapper
            return wrapper
        except ModuleNotFoundError:
            return None
