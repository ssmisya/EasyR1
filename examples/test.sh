#!/bin/bash

# set -x # 打印所有运行的命令

# # 检查是否在tmux会话内运行
# if [ -z "$TMUX" ]; then
#   # 创建一个新的tmux会话并执行此脚本
#   SESSION_NAME="tool_grpo_$(date +%Y%m%d_%H%M%S)"
#   echo "创建新的tmux会话: $SESSION_NAME"
#   tmux new-session -d -s "$SESSION_NAME" "bash $0 inside_tmux"
#   echo "tmux会话已在后台启动，你可以通过以下命令查看:"
#   echo "tmux attach -t $SESSION_NAME"
#   exit 0
# fi

# # 如果参数不是inside_tmux，说明是tmux内部调用，正常执行脚本
# if [ "$1" != "inside_tmux" ]; then
#   echo "请通过不带参数的方式运行此脚本，以启动tmux会话"
#   exit 1
# fi

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

export HF_ENDPOINT=https://hf-mirror.com
unset HF_ENDPOINT

code_base=/mnt/petrelfs/sunhaoyu/visual-code/EasyR1/examples
cd $code_base

# 生成带时间戳的日志文件名
log_file="/mnt/petrelfs/sunhaoyu/visual-code/EasyR1/scripts/logs/tool_grpo_$(date +%Y%m%d_%H%M%S).log"

export API_TYPE=openai
export OPENAI_API_URL=https://api.datapipe.app/v1/chat/completions
export OPENAI_API_KEY=sk-B3bRcR0fLubdoSmJ2cE13e57708c439aA14f825eB5Eb25De


# config_file=$1
export NCCL_DEBUG=ERROR
export NCCL_DEBUG_SUBSYS=ALL
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD=spawn

export PYTHONUNBUFFERED=1

# 清除可能影响Ray集群连接的环境变量
unset RAY_ADDRESS
unset RAY_REDIS_ADDRESS
# 强制Ray使用本地模式
export RAY_LOCAL_MODE=1
# 禁用Ray自动发现和连接
export RAY_DISABLE_AUTO_INIT=1
export RAY_DISABLE_AUTO_CONNECT=1
# 强制使用本地IP
export RAY_NODE_IP_ADDRESS=127.0.0.1

# 尝试停止任何现有Ray集群
ray stop || true

# 创建临时Ray配置文件
cat > /tmp/ray_config.yaml << EOF
cluster_name: local_cluster
max_workers: 0
provider:
  type: local
  head_ip: 127.0.0.1
EOF

# 设置Ray配置文件路径
export RAY_CONFIG_DIR=/tmp

gpus=0
cpus=2
quotatype="spot"
# quotatype="reserved"
config_file="/mnt/petrelfs/sunhaoyu/visual-code/EasyR1/examples/configs/tool.yaml"
MODEL_PATH=/mnt/petrelfs/sunhaoyu/visual-code/llm_weights/Qwen2.5-VL-3B-Instruct  # replace it with your local file path
OMP_NUM_THREADS=8 srun --partition=ai_moe --job-name="tool_grpo" --mpi=pmi2 --export=ALL --no-kill --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python \
-m test


