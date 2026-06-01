import importlib


def load(name: str, fw: int, fh: int):
    mod = importlib.import_module(f"jutsu.{name}")
    return mod.create(fw, fh)


def init_audio(name: str) -> None:
    mod = importlib.import_module(f"jutsu.{name}")
    fn  = getattr(mod, 'init_audio', None)
    if fn:
        fn()
