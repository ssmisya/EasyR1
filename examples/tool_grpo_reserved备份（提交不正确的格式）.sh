#!/bin/bash

# set -x # 打印所有运行的命令

# 检查是否在tmux会话内运行
if [ -z "$TMUX" ]; then
  # 创建一个新的tmux会话并执行此脚本
  SESSION_NAME="tool_grpo_resered_$(date +%Y%m%d_%H%M%S)"
  echo "创建新的tmux会话: $SESSION_NAME"
  tmux new-session -d -s "$SESSION_NAME" "bash $0 inside_tmux"
  echo "tmux会话已在后台启动，你可以通过以下命令查看:"
  echo "tmux attach -t $SESSION_NAME"
  exit 0
fi

# 如果参数不是inside_tmux，说明是tmux内部调用，正常执行脚本
if [ "$1" != "inside_tmux" ]; then
  echo "请通过不带参数的方式运行此脚本，以启动tmux会话"
  exit 1
fi



source ~/.bashrc
source ~/miniconda3/bin/activate visual

export http_proxy="http://sunhaoyu:Td2EgE1Vb5lBofIRcv6aiAHtwN2BvPFJlhTrYUdIvMWDeZ7rPq5jkRa4i2Qw@10.1.20.50:23128/"
export https_proxy="http://sunhaoyu:Td2EgE1Vb5lBofIRcv6aiAHtwN2BvPFJlhTrYUdIvMWDeZ7rPq5jkRa4i2Qw@10.1.20.50:23128/"
export HTTP_PROXY="http://sunhaoyu:Td2EgE1Vb5lBofIRcv6aiAHtwN2BvPFJlhTrYUdIvMWDeZ7rPq5jkRa4i2Qw@10.1.20.50:23128/"
export HTTPS_PROXY="http://sunhaoyu:Td2EgE1Vb5lBofIRcv6aiAHtwN2BvPFJlhTrYUdIvMWDeZ7rPq5jkRa4i2Qw@10.1.20.50:23128/"
export ALL_PROXY="http://sunhaoyu:Td2EgE1Vb5lBofIRcv6aiAHtwN2BvPFJlhTrYUdIvMWDeZ7rPq5jkRa4i2Qw@10.1.20.50:23128/"

export CUDA_HOME=/mnt/petrelfs/share/cuda-11.8
export PATH=/mnt/petrelfs/share/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-11.8/lib64:$LD_LIBRARY_PATH

export LDFLAGS="-ldl"

export CFLAGS="-I/mnt/petrelfs/sunhaoyu/visual-code/libaio/usr/include $CFLAGS"
export LDFLAGS="-L/mnt/petrelfs/sunhaoyu/visual-code/libaio/usr/lib $LDFLAGS"
export C_INCLUDE_PATH=/mnt/petrelfs/sunhaoyu/visual-code/libaio/usr/include
export LD_LIBRARY_PATH="/mnt/petrelfs/sunhaoyu/visual-code/libaio/usr/lib:$LD_LIBRARY_PATH"

# PyTorch CUDA 内存配置，解决内存碎片问题
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HF_ENDPOINT=https://hf-mirror.com
unset HF_ENDPOINT

code_base=/mnt/petrelfs/sunhaoyu/visual-code/EasyR1/examples
cd $code_base

# 生成带时间戳的日志文件名
log_dir="/mnt/petrelfs/sunhaoyu/visual-code/EasyR1/scripts/logs"
mkdir -p $log_dir
log_file="${log_dir}/tool_grpo_reserved_$(date +%Y%m%d_%H%M%S).log"


# config_file=$1
export NCCL_DEBUG=ERROR
export NCCL_DEBUG_SUBSYS=ALL

# export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD=spawn

export PYTHONUNBUFFERED=1

unset RAY_ADDRESS # 确保先清除，避免旧值干扰
unset RAY_REDIS_ADDRESS
# 确保你的 Ray 集群已在 10.140.37.138:6312 运行，Dashboard 在 10.140.37.138:8212
export RAY_ADDRESS="http://10.140.37.138:8212" # Ray Job CLI 和一些客户端会用此连接


quotatype="reserved"
config_file="/mnt/petrelfs/sunhaoyu/visual-code/EasyR1/examples/configs/tool_reserved.yaml"
MODEL_PATH=/mnt/petrelfs/sunhaoyu/visual-code/llm_weights/Qwen2.5-VL-3B-Instruct
checkpoint_dir="/mnt/petrelfs/sunhaoyu/visual-code/EasyR1/examples/checkpoints_reserved/easy_r1/qwen2_5_3b_tool_grpo"

# 函数：查找最新的检查点
find_latest_checkpoint() {
  if [ ! -d "$checkpoint_dir" ]; then
    echo ""
    return
  fi
  
  latest_checkpoint=$(find "$checkpoint_dir" -type d -name "global_step_*" | sort -V | tail -n 1)
  echo "$latest_checkpoint"
}

# 尝试停止任何现有Ray集群
ray stop || true


# 查找最新的检查点
latest_checkpoint=$(find_latest_checkpoint)



gpus=0
cpus=2
node_list="SH-IDC1-10-140-37-138"
export CUDA_VISIBLE_DEVICES=2,3,4,5

# 构建命令
cmd="OMP_NUM_THREADS=8 srun --partition=ai_moe -w ${node_list} --job-name=\"tool_grpo_reserved\" --mpi=pmi2 --export=ALL --no-kill --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype} \
python \
-m verl.trainer.main config=${config_file} worker.actor.model.model_path=${MODEL_PATH} ray_address=\"10.140.37.138:6312\""

# 如果存在最新检查点，则添加加载检查点的参数
if [ -n "$latest_checkpoint" ]; then
  echo "从检查点恢复训练: $latest_checkpoint"
  cmd="${cmd} trainer.load_checkpoint_path=${latest_checkpoint}"
else
  echo "从头开始训练"
fi

# 执行命令并记录日志
echo "开始训练" | tee -a ${log_file}
echo "$cmd" | tee -a ${log_file}
eval "$cmd" 2>&1 | tee -a ${log_file}

echo "运行日志已保存到: ${log_file}"


