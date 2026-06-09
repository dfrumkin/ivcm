from omegaconf import OmegaConf

__version__ = "0.1.0"

# Single source of truth for the class count is the ``class_names`` list; configs derive
# the count with ``${len:${class_names}}``. Registering at import time means ``train``,
# ``cross-validate``, and any test that imports ``ivcm`` have the resolver before
# composing/instantiating. ``replace=True`` keeps repeated imports (e.g. in tests) safe.
OmegaConf.register_new_resolver("len", len, replace=True)
