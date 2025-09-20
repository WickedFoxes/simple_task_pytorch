from collections import defaultdict

_REGISTRY = defaultdict(dict)

def register(kind: str, name: str):
    def deco(obj):
        _REGISTRY[kind][name.lower()] = obj
        return obj
    return deco

def build(kind: str, name: str, **kwargs):
    try:
        cls_or_fn = _REGISTRY[kind][name.lower()]
    except KeyError:
        raise ValueError(f"[Registry] {kind}:{name} is not registered.")
    return cls_or_fn(**kwargs) if callable(cls_or_fn) else cls_or_fn