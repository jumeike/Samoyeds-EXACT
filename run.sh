#!/bin/bash
# GPU job launcher for Samoyeds-EXACT

show_help() {
cat << EOF
Usage: ./run.sh [OPTIONS]

Options:
  -t, --time TIME       Job duration (default: 00:20:00)
  -g, --gpu GPU         GPU type (default: rtxa6000)
  -n, --ngpu N          Number of GPUs (default: 2)
  -c, --cpu N           CPUs per task (default: 4)
  -m, --mem SIZE        Memory (default: 64gb)
  -h, --help            Show this help

Available GPUs:
  l40s, rtx6000ada, h100-nvl, h100-sxm, h200-sxm, a100,
  rtx3070, rtx3090, rtxa4000, rtxa5000, rtxa6000

Examples:
  ./run.sh
  ./run.sh -t 2:00:00 -g rtx3090 -n 4
  ./run.sh --cpu 8 --mem 128gb --gpu a100 --ngpu 1
EOF
}

# Defaults
TIME="08:00:00"
GPU="rtxa6000"
NGPU=1
CPU=4
MEM="64gb"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -t|--time) TIME="$2"; shift 2 ;;
    -g|--gpu) GPU="$2"; shift 2 ;;
    -n|--ngpu) NGPU="$2"; shift 2 ;;
    -c|--cpu) CPU="$2"; shift 2 ;;
    -m|--mem) MEM="$2"; shift 2 ;;
    -h|--help) show_help; exit 0 ;;
    *) echo "Unknown option: $1"; show_help; exit 1 ;;
  esac
done

srun --pty \
  --cpus-per-task=${CPU} \
  --mem=${MEM} \
  --gres=gpu:${GPU}:${NGPU} \
  --partition=scavenger \
  --account=scavenger \
  --time=${TIME} \
  bash
