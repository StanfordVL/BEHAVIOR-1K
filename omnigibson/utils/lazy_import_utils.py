# Copyright (c) 2020, 2021 The HuggingFace Team
# Copyright (c) 2021 Philip May, Deutsche Telekom AG
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lazy-Imports module.

This is code taken from the `HuggingFace team <https://huggingface.co/>`__.
Many thanks to HuggingFace for
`your consent <https://github.com/huggingface/transformers/issues/12861#issuecomment-886712209>`__
to publish it as a standalone package.
"""

import importlib
import os
from types import ModuleType
from typing import Any


class LazyImporter(ModuleType):
    """Do lazy imports."""

    # Very heavily inspired by optuna.integration._IntegrationModule
    # https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
    def __init__(self, module_name, module):
        super().__init__("lazy_" + module_name)
        # self._modules = set(import_structure.keys())
        # self._class_to_module = {}
        # for key, values in import_structure.items():
        #     for value in values:
        #         self._class_to_module[value] = key
        self._module_path = module_name
        self._module = module
        self._submodules = {}

    def __getattr__(self, name: str) -> Any:
        # First, try the argument as a module name.
        submodule = self._get_module(name)
        if submodule:
            return submodule
        else:
            # If it's not a module name, try it as a member of this module.
            try:
                return getattr(self._module, name)
            except:
                raise AttributeError(
                    f"module {self.__name__} has no attribute {name}"
                ) from None

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