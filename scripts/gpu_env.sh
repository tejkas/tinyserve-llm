# Sourced by the GPU scripts: every large or growing directory lives in scratch, because the home
# directory is close to its 30 GB quota.
export SCRATCH_DIR="$HOME/scratch"
export VENV="$SCRATCH_DIR/venvs/tinyserve-gpu"
export UV_CACHE_DIR="$SCRATCH_DIR/.cache/uv"
export PIP_CACHE_DIR="$SCRATCH_DIR/.cache/pip"
export HF_HOME="$SCRATCH_DIR/.cache/huggingface"          # model weights land here
export TRITON_CACHE_DIR="$SCRATCH_DIR/.cache/triton"       # compiled Triton kernels
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH_DIR/.cache/inductor"
export PATH="$HOME/.local/bin:$PATH"                       # where uv's installer puts uv
