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

export HF_ENDPOINT=https://hf-mirror.com
unset HF_ENDPOINT

code_base=/mnt/petrelfs/songmingyang/code/reasoning/EasyR1
cd $code_base

export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_DISABLE_COMPILE_CACHE=1
export TMPDIR="/tmp/smy1"
export HYDRA_FULL_ERROR=1

export SLURM_JOB_ID=4964014
unset SLURM_JOB_ID

gpus=0
cpus=2
quotatype="reserved"
cluster_addr=SH-IDCA1404-10-140-54-22
model_path=/mnt/petrelfs/songmingyang/songmingyang/runs/tool_factory/tool_sft/Correct-Qwen2.5-VL-3B-Instruct-ToolSFT-GemmaReachQA
tool_log=/mnt/petrelfs/songmingyang/code/reasoning/EasyR1/scripts/logs/sft_rl_gemma_reach.jsonl
ckpt_path=/mnt/petrelfs/songmingyang/songmingyang/runs/tool_factory/gemma_reachqa/rl_listreward/sft_rl
experiment_name=Qwen2.5-VL-3B-Instruct-SFT-RL-GemmaReach

rm ${tool_log}

srun --partition=MoE --job-name="RL2" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
-w ${cluster_addr} \
ray job submit --address="http://$cluster_addr:8266" --working-dir ./examples \
-- python3 -m verl.trainer.main \
    config=configs/tool.yaml \
    worker.actor.model.model_path=${model_path} \
    worker.rollout.tool_log_file=${tool_log} \
    trainer.experiment_name=${experiment_name} \
    trainer.save_checkpoint_path=${ckpt_path} \
    trainer.total_epochs=50 \

# trainer.load_checkpoint_path=${ckpt_path}/global_step_40/ \