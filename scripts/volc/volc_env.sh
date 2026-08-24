# Shared vePFS locations for SpecForge volc ml_task runs, sourced by
# volc_prepare_env.sh and volc_entrypoint.sh.
#
# A task container's own filesystem is discarded when the task ends, and only
# the vePFS mount is shared between the dev instance and task instances. So uv,
# its managed CPython, and the project venv all have to live under
# TOOLCHAIN_DIR (shared with DeepSpec; SpecForge gets its own venv there).

export REPO_DIR=${REPO_DIR:-/root/songxianzheng/jiaomingyang/code/SpecForge}
export TOOLCHAIN_DIR=${TOOLCHAIN_DIR:-/root/songxianzheng/jiaomingyang/toolchain}
export PERSIST_HOME=${PERSIST_HOME:-/root/songxianzheng/jiaomingyang/home}

export PATH="${TOOLCHAIN_DIR}/bin:${PATH}"
export UV_PYTHON_INSTALL_DIR="${TOOLCHAIN_DIR}/uv-python"
export UV_CACHE_DIR="${TOOLCHAIN_DIR}/uv-cache"
export UV_PROJECT_ENVIRONMENT="${TOOLCHAIN_DIR}/venvs/specforge"
# The container's system pythons live outside vePFS, so uv must use one it
# manages itself under UV_PYTHON_INSTALL_DIR.
export UV_PYTHON_PREFERENCE=only-managed
export UV_LINK_MODE=copy
# The volces mirror is too stale for sglang 0.5.14's exact pins (flash-attn-4,
# sglang-kernel, sgl-deep-gemm, tokenspeed-*, nvidia-cutlass-dsl 4.5.2, ...),
# and direct pypi.org egress is throttled. The Tencent mirror carries every
# pin and serves ~10MB/s from here. Installation happens only on the dev
# instance; task nodes reuse the finished venv and never hit an index.
export UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX:-https://mirrors.cloud.tencent.com/pypi/simple/}
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-600}

# CUDA forward-compat user-space driver libs (libcuda shim and friends).
# Cluster nodes run driver R535 (CUDA 12), while the venv is the cu130 wheel
# line that sglang 0.5.14 requires; A800s are datacenter GPUs on an LTS
# branch, so the compat package bridges the gap.
export CUDA_COMPAT_DIR=${CUDA_COMPAT_DIR:-${TOOLCHAIN_DIR}/cuda-compat/13.3}
