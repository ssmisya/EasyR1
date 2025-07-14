# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import ray
from omegaconf import OmegaConf
import wandb

from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker # FSDP中涉及到了vllmrollout
from ..workers.reward import BatchFunctionRewardManager, SequentialFunctionRewardManager, ToolBatchFunctionRewardManager
from .config import PPOConfig
from .data_loader import create_dataloader
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role



# please make sure main_task is not scheduled on head
@ray.remote(num_cpus=1)
class Runner:
    """A runner for RL training."""

    def run(self, config: PPOConfig):
        # print config
        print(json.dumps(config.to_dict(), indent=2))
        # instantiate tokenizer
        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        # define worker classes
        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(FSDPWorker),
            Role.Critic: ray.remote(FSDPWorker),
            Role.RefPolicy: ray.remote(FSDPWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        if config.worker.reward.reward_type == "sequential":
            RewardManager = SequentialFunctionRewardManager
        elif config.worker.reward.reward_type == "batch":
            RewardManager = BatchFunctionRewardManager
        # 是tool_batch
        elif config.worker.reward.reward_type == "tool_batch":
            RewardManager = ToolBatchFunctionRewardManager
        else:
            raise NotImplementedError(f"Unknown reward type {config.worker.reward.reward_type}.")

        RemoteRewardManager = ray.remote(RewardManager).options(num_cpus=config.worker.reward.num_cpus)
        reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)
        val_reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)

        train_dataloader, val_dataloader = create_dataloader(config.data, tokenizer, processor)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


def main():
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(PPOConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    # 添加ray_address参数
    if "ray_address" in cli_args:
        ray_address = cli_args.pop("ray_address")
        # 将字符串"None"转换为Python的None
        if ray_address == "None":
            ray_address = None
    else:
        ray_address = None
        
    # 添加disable_ray参数，用于完全禁用Ray的分布式功能
    if "disable_ray" in cli_args:
        disable_ray = cli_args.pop("disable_ray")
        if isinstance(disable_ray, str):
            disable_ray = disable_ray.lower() == "true"
    else:
        disable_ray = False

    ppo_config = OmegaConf.merge(default_config, cli_args)
    ppo_config: PPOConfig = OmegaConf.to_object(ppo_config)
    ppo_config.deep_post_init()

    if disable_ray:
        print("Ray分布式功能已禁用，使用单进程模式")
        # 在这里可以添加单进程模式的代码
        # ...
        raise NotImplementedError("单进程模式尚未实现")
    elif not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                "PYTHONUNBUFFERED": "1",
                "RAY_DEBUG": "1",
            }
        }
        # 强制停止任何现有Ray集群并创建新的本地集群
        import os
        print(f"当前RAY_ADDRESS环境变量: {os.environ.get('RAY_ADDRESS', '未设置')}")
        print(f"当前RAY_REDIS_ADDRESS环境变量: {os.environ.get('RAY_REDIS_ADDRESS', '未设置')}")
        print(f"当前RAY_LOCAL_MODE环境变量: {os.environ.get('RAY_LOCAL_MODE', '未设置')}")
        print(f"ray_address参数值: {ray_address}, 类型: {type(ray_address)}")
        
        os.environ["RAY_ADDRESS"] = ""  # 清除可能存在的RAY_ADDRESS环境变量
        try:
            ray.shutdown()  # 尝试关闭任何现有Ray集群
            print("成功关闭现有Ray集群")
        except Exception as e:
            print(f"关闭Ray集群时出错: {e}")
        
        # 使用命令行参数中的ray_address
        # 如果ray_address为None，则不传递address参数
        try:
            if ray_address is None:
                print("初始化新的本地Ray集群(不指定address)")
                # 禁用Ray的自动发现功能，强制使用本地模式
                os.environ["RAY_DISABLE_AUTO_INIT"] = "1"
                os.environ["RAY_DISABLE_AUTO_CONNECT"] = "1"
                ray.init(
                    local_mode=True,  # 强制本地模式
                    ignore_reinit_error=True,
                    _redis_password=None,
                    include_dashboard=True,
                    runtime_env=runtime_env,
                    _node_ip_address="127.0.0.1",  # 强制使用本地回环地址
                    # _redis_max_memory=None,  # 禁用Redis内存限制
                    _plasma_directory=None,  # 使用默认临时目录
                    _memory=None,  # 不限制内存使用
                    _temp_dir=None,  # 使用默认临时目录
                    _system_config={"automatic_object_spilling_enabled": False}  # 禁用对象溢出
                )  # 强制创建新的本地集群
            elif ray_address == "":
                print("初始化新的本地Ray集群(空字符串address)")
                os.environ["RAY_DISABLE_AUTO_INIT"] = "1"
                os.environ["RAY_DISABLE_AUTO_CONNECT"] = "1"
                ray.init(
                    address="local",  # 使用local而不是空字符串
                    local_mode=True,  # 强制本地模式
                    ignore_reinit_error=True,
                    _redis_password=None,
                    include_dashboard=True,
                    runtime_env=runtime_env,
                    _node_ip_address="127.0.0.1",  # 强制使用本地回环地址
                    # _redis_max_memory=None,  # 禁用Redis内存限制
                    _plasma_directory=None,  # 使用默认临时目录
                    _memory=None,  # 不限制内存使用
                    _temp_dir=None,  # 使用默认临时目录
                    _system_config={"automatic_object_spilling_enabled": False}  # 禁用对象溢出
                )
            else:
                print(f"连接到现有Ray集群: {ray_address}")
                ray.init(address=ray_address, ignore_reinit_error=True, _redis_password=None, include_dashboard=True, runtime_env=runtime_env)
            print("Ray初始化成功")
        except Exception as e:
            print(f"Ray初始化失败: {e}")
            raise

    runner = Runner.remote()
    ray.get(runner.run.remote(ppo_config))


if __name__ == "__main__":
    main()