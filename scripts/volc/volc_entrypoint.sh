#!/usr/bin/env bash
# Entrypoint for volc ml_task instances running the online DSpark recipe:
#
#   examples/configs/qwen3.6-27b-dspark-online.yaml
#
# The online config uses external-services mode, so this script owns the two
# services the recipe expects on localhost before handing off to the unified
# trainer: mooncake_master (ports from the YAML) and one patched SGLang
# capture server (DSpark contract for Qwen3.6-27B). specforge train then
# supervises its own producer + consumer.
#
# Overridable via task Envs:
#   CONFIG_PATH        run config (default: the online DSpark recipe)
#   TARGET_MODEL_PATH  local target weights; the HF id in the YAML is not
#                      resolvable from task nodes without egress
#   TRAIN_DATA_PATH    conversation JSONL; the YAML's nemotron_v2 file does not
#                      exist on vePFS, so this defaults to the DeepSpec
#                      Qwen3.6-27B regenerated dataset
#   SGLANG_GPUS / TRAINER_GPUS  device split (defaults: 0 / 1)
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${script_dir}/volc_env.sh"

config_path=${CONFIG_PATH:-examples/configs/qwen3.6-27b-dspark-online.yaml}
target_model_path=${TARGET_MODEL_PATH:-/root/songxianzheng/models/Qwen3.6-27B}
train_data_path=${TRAIN_DATA_PATH:-/root/songxianzheng/jiaomingyang/code/DeepSpec/train_datasets/qwen36_27b/perfectblend_train_regen.jsonl}
sglang_gpus=${SGLANG_GPUS:-0}
trainer_gpus=${TRAINER_GPUS:-1}
sglang_port=${SGLANG_PORT:-30000}

# HF caches and anything else keyed off ~ must land on vePFS, not the
# ephemeral container rootfs.
export HOME="${PERSIST_HOME}"
mkdir -p "${HOME}"

venv="${UV_PROJECT_ENVIRONMENT}"
if [[ ! -x "${venv}/bin/python" ]]; then
    echo "no venv at ${venv}; run scripts/volc/volc_prepare_env.sh first" >&2
    exit 1
fi
export PATH="${venv}/bin:${PATH}"

# The venv carries the cu130 torch wheel line, which needs an r580+ driver.
# Cluster nodes run R535, so bridge with the CUDA forward-compat libs staged
# on vePFS by volc_prepare_env.sh. No `| head` here: under pipefail, head's
# early close makes nvidia-smi die with SIGPIPE and takes the script down.
driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
driver_version=${driver_version%%$'\n'*}
echo "nvidia driver ${driver_version}; gpus: $(nvidia-smi -L | wc -l)"
if (( ${driver_version%%.*} < 580 )); then
    if [[ ! -e "${CUDA_COMPAT_DIR}/libcuda.so.1" ]]; then
        echo "driver ${driver_version} needs the CUDA compat libs, but" >&2
        echo "${CUDA_COMPAT_DIR} is missing; run volc_prepare_env.sh" >&2
        exit 1
    fi
    export LD_LIBRARY_PATH="${CUDA_COMPAT_DIR}:${LD_LIBRARY_PATH:-}"
    echo "using CUDA forward-compat libs at ${CUDA_COMPAT_DIR}"
fi

# mooncake_master links libcudart.so.13 dynamically but has no RPATH into the
# venv's pip-provided CUDA runtime, so expose it process-wide.
export LD_LIBRARY_PATH="${venv}/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

python - <<'PY'
import torch

assert torch.cuda.is_available(), "torch cannot reach CUDA under this driver"
print("cuda ok:", torch.cuda.get_device_name(0), "runtime", torch.version.cuda)
PY

[[ -f "${target_model_path}/config.json" ]] \
    || { echo "target model missing: ${target_model_path}" >&2; exit 1; }
[[ -f "${train_data_path}" ]] \
    || { echo "train data missing: ${train_data_path}" >&2; exit 1; }

cd "${REPO_DIR}"

# A fresh disaggregated attempt requires fresh control and consumer-state
# directories, so scope both to this task attempt. Consumer state (SQLite/WAL)
# stays node-local; control/logs persist on vePFS.
run_id="qwen3.6-27b-dspark-online"
attempt_id=${ATTEMPT_ID:-${MLP_TASK_ID:-manual}-$(date +%Y%m%d%H%M%S)}
output_dir="outputs/${run_id}"
control_dir="${output_dir}/control-${attempt_id}"
consumer_state_dir="/tmp/specforge/${run_id}-${attempt_id}/consumer-state"
log_dir="${control_dir}/logs"
mkdir -p "${log_dir}" "${consumer_state_dir}"

echo "attempt=${attempt_id} config=${config_path}"
echo "model=${target_model_path}"
echo "data=${train_data_path}"
echo "control_dir=${control_dir}"

service_pids=()
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    local pid
    for pid in "${service_pids[@]}"; do
        kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
    done
    exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# accept_any=true treats any HTTP response as ready (mooncake's metadata
# endpoint answers 404 for a probe key, which still proves the server is up).
wait_http() {
    local label=$1 url=$2 pid=$3 log=$4 attempts=$5 accept_any=${6:-false}
    local attempt curl_flags
    for ((attempt = 1; attempt <= attempts; attempt++)); do
        kill -0 "${pid}" 2>/dev/null \
            || { echo "${label} exited during startup; see ${log}" >&2; exit 1; }
        if [[ "${accept_any}" == true ]]; then
            curl_flags=(-sS)
        else
            curl_flags=(-fsS)
        fi
        if curl "${curl_flags[@]}" --max-time 2 -o /dev/null "${url}" 2>/dev/null; then
            echo "${label} ready at ${url}"
            return 0
        fi
        sleep 2
    done
    echo "${label} not ready at ${url} after $((attempts * 2))s; see ${log}" >&2
    exit 1
}

# Ports below mirror deployment.disaggregated.* in the online YAML.
CUDA_VISIBLE_DEVICES="" setsid mooncake_master \
    --enable_http_metadata_server=true \
    --http_metadata_server_host=127.0.0.1 \
    --rpc_port=35551 \
    --http_metadata_server_port=35880 \
    --metrics_port=35903 \
    > "${log_dir}/mooncake.log" 2>&1 &
service_pids+=($!)
wait_http mooncake "http://127.0.0.1:35880/metadata?key=specforge-health-check" \
    "${service_pids[-1]}" "${log_dir}/mooncake.log" 60 true

# DSpark capture contract for Qwen3.6-27B: method dspark, aux layer ids from
# configs/qwen3.6-27b-dspark.json target_layer_ids. Context length is
# data.max_length + 7 capture headroom; tp1 / mem 0.7 mirror the managed_local
# sibling recipe qwen3.6-27b-dspark-disaggregated.yaml.
CUDA_VISIBLE_DEVICES="${sglang_gpus}" \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
MOONCAKE_METADATA_SERVER=http://127.0.0.1:35880/metadata \
MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:35551 \
MOONCAKE_LOCAL_HOSTNAME=127.0.0.1 \
MOONCAKE_PROTOCOL=tcp \
MOONCAKE_GLOBAL_SEGMENT_SIZE=68719476736 \
MOONCAKE_LOCAL_BUFFER_SIZE=1073741824 \
setsid python -m sglang.launch_server \
    --model-path "${target_model_path}" \
    --dtype bfloat16 \
    --trust-remote-code \
    --skip-tokenizer-init \
    --tp-size 1 \
    --chunked-prefill-size -1 \
    --enable-spec-capture \
    --spec-capture-method dspark \
    --spec-capture-aux-layer-ids 2 17 32 47 62 \
    --attention-backend flashinfer \
    --mem-fraction-static 0.7 \
    --disable-radix-cache \
    --context-length 4103 \
    --host 127.0.0.1 \
    --port "${sglang_port}" \
    > "${log_dir}/capture-server.log" 2>&1 &
service_pids+=($!)
# 27B bf16 weights come off vePFS; allow up to 30 minutes.
wait_http capture-server "http://127.0.0.1:${sglang_port}/health" \
    "${service_pids[-1]}" "${log_dir}/capture-server.log" 900

# Both producer in-flight watermarks must reach the consumer's optimizer-step
# quantum (consumer_world_size * batch_size * accumulation_steps = 512 for
# this recipe); the 256/192 defaults are rejected at startup.
CUDA_VISIBLE_DEVICES="${trainer_gpus}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
specforge train -c "${config_path}" \
    "model.target_model_path=${target_model_path}" \
    "data.train_data_path=${train_data_path}" \
    "output_dir=${output_dir}" \
    "deployment.disaggregated.control_dir=${control_dir}" \
    "deployment.disaggregated.consumer_state_dir=${consumer_state_dir}" \
    "deployment.disaggregated.server_urls=[\"http://127.0.0.1:${sglang_port}\"]" \
    "runtime.in_flight_high_watermark=${IN_FLIGHT_HIGH_WATERMARK:-640}" \
    "runtime.in_flight_low_watermark=${IN_FLIGHT_LOW_WATERMARK:-512}"
