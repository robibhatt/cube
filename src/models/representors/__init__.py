"""Model representor package bootstrap."""

from src.utils.plugin_loader import import_submodules

from .representor_registry import REPRESENTOR_REGISTRY, register_representor

# Import all submodules so their registration decorators run and populate
# :data:`REPRESENTOR_REGISTRY`.  During partial initialisation (e.g. when
# imported from modules that themselves depend on representor utilities) some
# submodules may trigger circular imports.  We therefore swallow such errors so
# that directly importing a specific representor remains possible.
try:  # pragma: no cover - defensive import
    import_submodules(__name__)
except Exception:
    pass

__all__ = ["REPRESENTOR_REGISTRY", "register_representor"]