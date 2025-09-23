"""Utility functions for importing plugin modules."""

from importlib import import_module
from pkgutil import walk_packages
from types import ModuleType
from typing import List

__all__ = ["import_submodules"]


def import_submodules(package_name: str) -> List[ModuleType]:
    """Import all submodules of ``package_name``.

    This function walks the specified package and imports every submodule. It
    is mainly used for loading modules that register themselves as plugins via
    import side effects.
    """
    package = import_module(package_name)
    modules: List[ModuleType] = [package]

    if hasattr(package, "__path__"):
        for info in walk_packages(package.__path__, package.__name__ + "."):
            modules.append(import_module(info.name))

    return modules

