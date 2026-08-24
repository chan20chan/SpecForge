#!/usr/bin/env bash
# One-time (idempotent) setup of the vePFS-resident Python environment that
# volc ml_task instances reuse. Run it from the dev instance; no GPU is needed
# since the CUDA wheels install without one.
#
#   bash scripts/volc/volc_prepare_env.sh
#
# Re-run it after changing pyproject.toml or the sglang capture patch.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${script_dir}/volc_env.sh"

mkdir -p "${TOOLCHAIN_DIR}/bin" "${UV_PYTHON_INSTALL_DIR}" "${UV_CACHE_DIR}" \
    "${PERSIST_HOME}"

if [[ ! -x "${TOOLCHAIN_DIR}/bin/uv" ]]; then
    curl -LsSf "https://astral.sh/uv/0.12.5/install.sh" \
        | env UV_INSTALL_DIR="${TOOLCHAIN_DIR}/bin" INSTALLER_NO_MODIFY_PATH=1 sh
fi

if [[ ! -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]]; then
    uv venv --python 3.11 "${UV_PROJECT_ENVIRONMENT}"
fi

# SpecForge has no uv.lock; install straight from pyproject. Mooncake is the
# disaggregated transport client; the resolved torch is the cu130 wheel line,
# so use the CUDA 13 mooncake variant per docs/basic_usage/disaggregated_training.md.
uv pip install --python "${UV_PROJECT_ENVIRONMENT}/bin/python" \
    -e "${REPO_DIR}" \
    mooncake-transfer-engine-cuda13

# Task nodes run driver R535 while this venv is the cu130 wheel line, so stage
# the CUDA forward-compat user-space libs (libcuda shim) on vePFS. The deb's
# postinst normally creates the .so.1 symlinks; recreate them here.
if [[ ! -e "${CUDA_COMPAT_DIR}/libcuda.so.1" ]]; then
    compat_deb="cuda-compat-13-3_610.57.04-1ubuntu1_amd64.deb"
    tmp_dir=$(mktemp -d)
    curl -sL --max-time 600 -o "${tmp_dir}/${compat_deb}" \
        "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/${compat_deb}"
    dpkg-deb -x "${tmp_dir}/${compat_deb}" "${tmp_dir}/extract"
    mkdir -p "${CUDA_COMPAT_DIR}"
    cp "${tmp_dir}/extract/usr/local/cuda-13.3/compat/"* "${CUDA_COMPAT_DIR}/"
    (
        cd "${CUDA_COMPAT_DIR}"
        for lib in *.so.610.57.04; do
            base=${lib%.610.57.04}
            ln -sf "${lib}" "${base}.1"
            ln -sf "${base}.1" "${base}"
        done
    )
    rm -rf "${tmp_dir}"
fi

# The online capture server runs from this same venv, so the installed sglang
# must carry the spec-capture patch (content-aware, idempotent).
PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}" \
    bash "${REPO_DIR}/scripts/apply_sglang_spec_capture_patch.sh" --target v0.5.14

echo "environment ready at ${UV_PROJECT_ENVIRONMENT}"
"${UV_PROJECT_ENVIRONMENT}/bin/python" - <<'PY'
import sglang
import torch

print("torch", torch.__version__)
print("sglang", sglang.__version__)
import mooncake  # noqa: F401

print("mooncake client import ok")
PY
