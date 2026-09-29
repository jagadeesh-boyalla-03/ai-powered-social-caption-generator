try:
    from .index import app
except Exception:
    from api.index import app
