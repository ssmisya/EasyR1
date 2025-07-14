source ~/.bashrc
source ~/miniconda3/bin/activate visual

# environment variables
export OMP_NUM_THREADS=4
# new_proxy_address=http://sunhaoyu:Td2EgE1Vb5lBofIRcv6aiAHtwN2BvPFJlhTrYUdIvMWDeZ7rPq5jkRa4i2Qw@10.1.20.50:23128/
# export http_proxy=$new_proxy_address
# export https_proxy=$new_proxy_address
# export HTTP_PROXY=$new_proxy_address
# export HTTPS_PROXY=$new_proxy_address

unset http_proxy
unset HTTP_PROXY
# unset https_proxy
# unset HTTPS_PROXY

code_base=/mnt/petrelfs/sunhaoyu/visual-code/EasyR1
cd $code_base


export VLLM_WORKER_MULTIPROC_METHOD="spawn"
# export VLLM_ATTENTION_BACKEND=XFORMERS
# mkdir -p /mnt/petrelfs/songmingyang/tmp/ray/log1
# ln -s /mnt/petrelfs/songmingyang/tmp /tmp/smy1
export TMPDIR="/mnt/petrelfs/sunhaoyu/visual-code/tmp1"
# unset TMPDIR
HYDRA_FULL_ERROR=1


# export SLURM_JOB_ID=5139207 # 原本没有注释

# unset SLURM_JOB_ID
export RAY_memory_monitor_refresh_ms=0
###
gpus=8
cpus=88
quotatype="spot"

###
cluster_addr=SH-IDC1-10-140-37-44
###
cluster_ip=10.140.37.44
###
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

port=6312
dashboard_port=8212
echo "Starting ray at IP: $cluster_ip, PORT: $port"
srun --partition=ai_moe \
-w ${cluster_addr} \
--job-name="RAY-CLUSTER" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
ray start --head --node-ip-address="$cluster_ip" --port=$port --dashboard-host=0.0.0.0 --dashboard-port=$dashboard_port \
--num-cpus ${cpus} --num-gpus ${gpus}  --block --temp-dir="$TMPDIR"