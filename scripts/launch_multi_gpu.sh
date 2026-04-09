#!/bin/bash
# Multi-GPU launch script for choochoo training
#
# Usage:
#   Single node, 4 GPUs:
#     ./scripts/launch_multi_gpu.sh --config examples/wan22_lora.yaml --nproc 4
#
#   Multi-node (run on each node):
#     RANK=0 MASTER_ADDR=192.168.1.100 ./scripts/launch_multi_gpu.sh \
#       --config examples/wan22_lora.yaml --nproc 8 --nnodes 2
#

set -euo pipefail

CONFIG=""
NPROC=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 1)
NNODES=1
NODE_RANK=0
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29500}
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2;;
        --nproc) NPROC="$2"; shift 2;;
        --nnodes) NNODES="$2"; shift 2;;
        --node-rank) NODE_RANK="$2"; shift 2;;
        --master-addr) MASTER_ADDR="$2"; shift 2;;
        --master-port) MASTER_PORT="$2"; shift 2;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "Error: --config is required"
    echo "Usage: $0 --config examples/wan22_lora.yaml [--nproc N] [--nnodes N]"
    exit 1
fi

echo "=== choochoo Multi-GPU Training ==="
echo "Config: $CONFIG"
echo "GPUs per node: $NPROC"
echo "Nodes: $NNODES"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "==================================="

torchrun \
    --nproc_per_node="$NPROC" \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    train.py \
    --config "$CONFIG" \
    $EXTRA_ARGS
