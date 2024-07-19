from huggingface_hub import snapshot_download
model_id="MLP-KTLim/llama-3-Korean-Bllossom-8B"
snapshot_download(repo_id=model_id, local_dir="llama-3-Korean-Bllossom-8B",
                  local_dir_use_symlinks=False, revision="main")
