import importlib
try:
    index_mod = importlib.import_module("api.index")
    app = index_mod.app
except Exception:
    from .index import app
