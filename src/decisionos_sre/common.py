import hashlib
import json
import os
from pathlib import Path
import random
import importlib.metadata as metadata

DATA_REV = "afeacb11bcc94dadfd1c8f483ee4377b2b8b614e"
MODEL_REV = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
MODEL_NAME = "answerdotai/ModernBERT-base"
SERIALIZER = "metrics-relative-v1"
ONTOLOGY = {"cpu": "cpu_stress", "mem": "memory_stress", "disk": "disk_io_stress",
            "delay": "network_delay", "loss": "network_packet_loss"}
FAULTS = list(ONTOLOGY.values())

def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    os.replace(tmp, path)

def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))

def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def environment():
    packages = ["torch", "transformers", "numpy", "pandas", "pyarrow", "pydantic", "fastapi", "scipy"]
    return {p: metadata.version(p) for p in packages}
