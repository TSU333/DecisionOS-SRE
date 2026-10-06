from pathlib import Path
from decisionos_sre.data import download
from decisionos_sre.common import MODEL_REV, file_hash, save
root=Path("artifacts/backbone")
files=["config.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json","README.md","model.safetensors"]
for name in files:
    download(f"https://huggingface.co/answerdotai/ModernBERT-base/resolve/{MODEL_REV}/{name}",root/name)
save(root/"manifest.json",{"revision":MODEL_REV,"files":{name:file_hash(root/name) for name in files}})
