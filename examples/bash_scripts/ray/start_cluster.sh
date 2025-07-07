source ~/.bashrc
source ~/anaconda3/bin/activate easyr1_tool

# environment variables
export OMP_NUM_THREADS=4
new_proxy_address=http://your_proxy_addr/
export http_proxy=$new_proxy_address
export https_proxy=$new_proxy_address
export HTTP_PROXY=$new_proxy_address
export HTTPS_PROXY=$new_proxy_address

unset http_proxy
unset HTTP_PROXY
# unset https_proxy
# unset HTTPS_PROXY

code_base=/mnt/petrelfs/songmingyang/code/reasoning/EasyR1
cd $code_base



job_id=4671281


export VLLM_WORKER_MULTIPROC_METHOD="spawn"
# export VLLM_ATTENTION_BACKEND=XFORMERS
# mkdir -p /mnt/petrelfs/songmingyang/tmp/ray/log1
# ln -s /mnt/petrelfs/songmingyang/tmp /tmp/smy1
export TMPDIR="/mnt/petrelfs/songmingyang/tmp1"
# unset TMPDIR
HYDRA_FULL_ERROR=1


export SLURM_JOB_ID=5139207
# unset SLURM_JOB_ID
export RAY_memory_monitor_refresh_ms=0
gpus=8
cpus=64
quotatype="reserved"

cluster_addr=SH-IDC1-10-140-37-35
cluster_ip=10.140.54.35

port=6381
dashboard_port=8267
echo "Starting ray at IP: $cluster_ip, PORT: $port"
srun --partition=ai_moe \
-w ${cluster_addr} \
--job-name="RAY-CLUSTER" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
ray start --head --node-ip-address="$cluster_ip" --port=$port --dashboard-host=0.0.0.0 --dashboard-port=$dashboard_port \
--num-cpus "64" --num-gpus "8"  --block --temp-dir="$TMPDIR"


# srun --partition=MoE --job-name="RL" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
# -w ${cluster_addr} \ 
# ray job submit --address="http://$cluster_addr:8266" --working-dir . \
# -- python3 -m verl.trainer.main_ppo \