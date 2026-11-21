import importlib.util
import json
from pathlib import Path


def load_config(path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location("experiment_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "CONFIG"):
        raise ValueError(f"{path} must define CONFIG.")
    return module.CONFIG


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"Saved: {path.resolve()}")


def print_config(config):
    print(json.dumps(config, ensure_ascii=False, indent=2))
