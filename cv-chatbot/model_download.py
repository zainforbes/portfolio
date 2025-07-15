# download_mistral.py
from huggingface_hub import snapshot_download
from pathlib import Path

out = Path(__file__).parent / "models" / "mistral-7b-instruct"
out.mkdir(parents=True, exist_ok=True)

# Fetch everything so we get config.json + generation_config.json + merges.txt + weights + tokenizer
snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    local_dir=out,
    use_auth_token=True
)

# Rename for HF
(cons := out / "consolidated.safetensors").rename(out / "model.safetensors")
(tok := out / "tokenizer.model.v3").rename(out / "tokenizer.model")
print("✅ Downloaded & prepared Mistral files in", out)
