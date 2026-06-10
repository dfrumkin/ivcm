from omegaconf import OmegaConf

__version__ = "0.1.0"

__all__ = ["register_resolvers"]


def register_resolvers() -> None:
    """Register the project's custom OmegaConf resolvers.

    Single source of truth for the class count is the ``class_names`` list; configs derive
    the count with ``${len:${class_names}}``. ``replace=True`` makes repeated calls safe, so
    this can be invoked from multiple entry points without raising.
    """
    OmegaConf.register_new_resolver("len", len, replace=True)


# Register at import time so ``train``, ``cross-validate``, and any production ``import
# ivcm.*`` have the resolver before composing/instantiating. Tests call
# ``register_resolvers()`` explicitly from ``conftest`` instead of relying on this.
register_resolvers()
