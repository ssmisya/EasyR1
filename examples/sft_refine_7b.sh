#!/bin/bash

# set -x # 打印所有运行的命令

# 检查是否在tmux会话内运行
if [ -z "$TMUX" ]; then
  # 创建一个新的tmux会话并执行此脚本
  ###
  SESSION_NAME="sft_refine_7b_step245_$(date +%Y%m%d_%H%M%S)"
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

###
export NO_PROXY="localhost,127.0.0.1,10.140.37.136,10.136.0.0/16,10.1.0.0/16"
export no_proxy="localhost,127.0.0.1,10.140.37.136,10.136.0.0/16,10.1.0.0/16"

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

code_base=/mnt/petrelfs/sunhaoyu/visual-code/EasyR1
# 将工作目录切换到 Ray Job 提交的上下文目录，通常是你的项目根目录或包含你主脚本的目录
# 这里设置为 code_base 的父目录，因为你的训练脚本 `main.py` 在 `EasyR1/examples` 下
# Ray Job 提交时，working-dir 应该指向 Ray 可以在其中找到你的代码的目录
project_root=/mnt/petrelfs/sunhaoyu/visual-code/EasyR1
cd $project_root

# 生成带时间戳的日志文件名
###
log_dir="/mnt/petrelfs/sunhaoyu/visual-code/EasyR1/scripts/logs"
mkdir -p $log_dir
log_file="${log_dir}/sft_refine_7b_step245_$(date +%Y%m%d_%H%M%S).log"


export NCCL_DEBUG=ERROR
export NCCL_DEBUG_SUBSYS=ALL

export VLLM_WORKER_MULTIPROC_METHOD=spawn

export PYTHONUNBUFFERED=1

unset RAY_ADDRESS # 确保先清除，避免旧值干扰
unset RAY_REDIS_ADDRESS
# 确保你的 Ray 集群已在 10.140.37.136:6312 运行，Dashboard 在 10.140.37.136:8212
# Ray Job CLI 连接的是 Dashboard 地址
###
export RAY_ADDRESS="http://10.140.37.136:8212"

###
quotatype="reserved"
### sft是sft_7b.yaml
config_file="examples/configs/sft_7b_245.yaml"
###
# MODEL_PATH=/mnt/petrelfs/sunhaoyu/visual-code/llm_weights/Qwen2.5-VL-3B-Instruct
MODEL_PATH=/mnt/petrelfs/share_data/songmingyang/runs/tool_factory/sft/v1/Qwen2.5-VL-7B-Instruct-ToolSFTv1/checkpoint-245
###
checkpoint_dir="/mnt/petrelfs/sunhaoyu/visual-code/EasyR1/checkpoints/checkpoints_sft7b_245_refine"

# 函数：查找最新的检查点
find_latest_checkpoint() {
  if [ ! -d "$checkpoint_dir" ]; then
    echo ""
    return
  fi
  
  latest_checkpoint=$(find "$checkpoint_dir" -type d -name "global_step_*" | sort -V | tail -n 1)
  echo "$latest_checkpoint"
}

# 尝试停止任何现有Ray集群 (这个通常用于开发或测试，在提交Job前不需要)
# ray stop || true


# 查找最新的检查点
latest_checkpoint=$(find_latest_checkpoint)

# 定义 Ray Job 要执行的 Python 命令
# 注意：这里不再需要 srun，Ray 会自己调度资源
###
python_command="python -m verl.trainer.main config=${config_file} worker.actor.model.model_path=${MODEL_PATH} ray_address=\"10.140.37.136:6312\""

# 如果存在最新检查点，则添加加载检查点的参数
if [ -n "$latest_checkpoint" ]; then
  echo "从检查点恢复训练: $latest_checkpoint"
  python_command="${python_command} trainer.load_checkpoint_path=${latest_checkpoint}"
else
  echo "从头开始训练"
fi

echo "开始通过 Ray Job 提交训练" | tee -a ${log_file}
echo "Ray Job Command: ray job submit --address=\"$RAY_ADDRESS\" --working-dir=\"$project_root\" -- python_command..." | tee -a ${log_file}

# 使用 ray job submit 提交任务
# --working-dir 指定了 Ray Job 的工作目录，Ray 会将此目录的内容同步到集群节点
# runtime-env要设置，为了防止把不需要的文件放进去
### 底下的ray_address 需要修改
ray job submit --address="$RAY_ADDRESS" \
  --working-dir . \
  --runtime-env ray_exclude.yaml \
  -- bash -c "${python_command}" \
  2>&1 | tee -a ${log_file}

echo "运行日志已保存到: ${log_file}"