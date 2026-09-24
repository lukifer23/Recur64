# Source before running recur64 on the HP machine (Git Bash). Process-local
# user-space CUDA 12.9.1; no system PATH, registry, or driver change.
export CUDA_PATH="$LOCALAPPDATA\Recur64\cuda\12.9.1"
export PATH="$(cygpath -u "$CUDA_PATH")/bin:$PATH"
