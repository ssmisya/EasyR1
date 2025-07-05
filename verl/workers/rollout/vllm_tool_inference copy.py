import re
import json
import torch
import signal
import time
import fcntl
import os
import datetime
import concurrent.futures  # 添加并发处理库

from vllm import LLM, SamplingParams
from typing import List, Union, Dict, Optional, Tuple
from tool_server.tool_workers.tool_manager.base_manager import ToolManager
from PIL import Image
from contextlib import contextmanager
from copy import deepcopy
from tqdm import tqdm as tqdm_rank0
import threading
from io import BytesIO
import base64

# from ....utils.utils import pil_to_base64, base64_to_pil, append_jsonl

# 添加统计数据类
class ToolCallStats:
    def __init__(self):
        self.total_calls = 0
        self.successful_calls = 0
        # 添加工具特定的统计
        self.tool_calls = {}  # 格式: {tool_name: total_calls}
        self.tool_success = {}  # 格式: {tool_name: successful_calls}
        # 定义要统计的工具列表
        self.tracked_tools = [
            "GroundingDINO","OCR","SegmentRegionAroundPoint","Point","Crop","DrawLine", "DrawShape", "HighlightBox", "MaskBox", "LanguageModel", "GetSubplotInfo", "GetBarInfo"
        ]
        
    def add_call(self, success: bool, tool_name: str = None):
        self.total_calls += 1
        if success:
            self.successful_calls += 1
        
        # 记录特定工具的调用
        if tool_name:
            
            # 检查工具是否在跟踪列表中，如果不在则归类为nonexistent_tool
            if tool_name not in self.tracked_tools:
                tool_name = "Nonexistent_tool"
            
            if tool_name not in self.tool_calls:
                self.tool_calls[tool_name] = 0
                self.tool_success[tool_name] = 0
            
            self.tool_calls[tool_name] += 1
            if success:
                self.tool_success[tool_name] += 1
            
    # def get_success_rate(self):
    #     if self.total_calls == 0:
    #         return 0.0
    #     return self.successful_calls / self.total_calls
    
    # def get_tool_success_rate(self, tool_name):
    #     if tool_name not in self.tool_calls or self.tool_calls[tool_name] == 0:
    #         return 0.0
    #     return self.tool_success[tool_name] / self.tool_calls[tool_name]
    
    def get_stats_dict(self):
        stats = {
            "tool_total_calls": self.total_calls,
            "tool_successful_calls": self.successful_calls,
            # "tool_success_rate": self.get_success_rate()
        }
        
        # 添加各个工具的统计数据
        for tool_name in self.tool_calls:
            stats[f"tool_{tool_name}_calls"] = self.tool_calls[tool_name]
            stats[f"tool_{tool_name}_success"] = self.tool_success[tool_name]
            # stats[f"tool_{tool_name}_success_rate"] = self.get_tool_success_rate(tool_name)
            
        return stats
    
    def reset(self):
        self.total_calls = 0
        self.successful_calls = 0
        self.tool_calls = {}
        self.tool_success = {}

# 定义超时异常类
class TimeoutException(Exception): pass

# 定义超时处理函数
def _timeout_handler(signum, frame):
    raise TimeoutException("chat() timed out.")
signal.signal(signal.SIGALRM, _timeout_handler)

_lock_lock = threading.Lock()
_file_locks = {}

def _get_file_lock(filepath):
    """Get a lock for a specific file path."""
    with _lock_lock:
        if filepath not in _file_locks:
            _file_locks[filepath] = threading.Lock()
        return _file_locks[filepath]

def append_jsonl(data, filename):
    '''
        追加数据到jsonl文件，线程安全
    '''
    with _get_file_lock(filename):
        with open(filename, 'a', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')

def pil_to_base64(img: Image.Image, url_format = False) -> str:
    """
    Convert a PIL image to a base64 encoded string.
    
    Args:
        img (Image.Image): The PIL image to convert.
        
    Returns:
        str: Base64 encoded string representation of the image.
    """
    buffered = BytesIO()
    # 确保图像是RGB模式，如果是RGBA或其他模式则转换
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        # 需要先转换为RGB模式
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)  # 使用alpha通道作为mask
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    if url_format:
        img_str = f"data:image/jpeg;base64,{img_str}"
    return img_str

def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))
    

def base64_to_pil(b64_str: str) -> Image.Image:
    """
    Convert a base64 encoded string into a PIL image.
    
    Args:
        b64_str (str): The base64 encoded image string.
        
    Returns:
        Image.Image: The resulting PIL image.
    """
    # Remove the data URI scheme if present
    if b64_str.startswith("data:image"):
        b64_str = b64_str.split("base64,")[-1]
    return load_image_from_base64(b64_str)



class VllmToolInferencer(object):
    """
    VLLM工具推理器类，用于处理模型推理和工具调用
    """
    def __init__(
        self,
        vllm_model: LLM = None,
        model_name: str = None,
        model_configs: dict = None,
        tool_controller_addr: str = None,
        batch_size: int = 1,
        model_mode: str = "general",
        max_rounds: int = 5,
        stop_token: str = "<stop>",
    ):
        """
        初始化VLLM工具推理器
        
        参数:
            vllm_model: 预加载的VLLM模型实例
            model_name: 模型名称
            model_configs: 模型配置参数
            tool_controller_addr: 工具控制器地址
            batch_size: 批处理大小
            model_mode: 模型模式，支持general和llava_plus
            max_rounds: 最大对话轮数
            stop_token: 停止标记
        """
        # 初始化VLLM模型
        # 如果有了vllm_model，则直接使用vllm_model，不需要model_name和model_configs
        if vllm_model is not None:
            self.vllm_model = vllm_model
        elif model_name is not None:
            if model_configs is not None:
                self.vllm_model = LLM(model=model_name, **model_configs)
            else:
                self.vllm_model = LLM(model=model_name)
        else:
            raise ValueError("Either vllm_model or model_name must be provided.")
        
        # 初始化工具管理器
        tool_manager = ToolManager(tool_controller_addr)
        tool_controller_addr_display = tool_controller_addr if tool_controller_addr else "Auto"
        print(f"controller_addr: {tool_controller_addr_display}")
        print(f"Avaliable tools are {tool_manager.available_tools}")
        
        self.tool_manager = tool_manager
        self.batch_size = batch_size
        self.model_mode = model_mode
        self.max_rounds = max_rounds
        self.stop_token = stop_token
        
        # 添加图像字典，用于存储每个对话项的图像
        self.image_history = {}
        
        # 添加工具调用统计
        self.tool_stats = ToolCallStats()
    
    def append_conversation_fn(
        self,
        conversation, 
        text: str, 
        image=None, 
        role: str = "user",
    ):
        """
        将新消息追加到对话历史中
        
        参数:
            conversation (list): 当前对话历史
            text (str): 要追加的文本消息
            image: (可选) 要包含的图像
            role (str): 发送者角色 (默认为 "user")
            
        返回:
            更新后的对话列表
        """
        if image:
            image_base64 = pil_to_base64(image, url_format=True)
            new_messages = [
                {
                    "role": role,
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_base64}
                        },
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
                }
            ]
        else:
            new_messages = [
                {
                    "role": role,
                    "content": [
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
                }
            ]
        
        conversation.extend(new_messages)
        return conversation

    def handle_tool_result(
        self,
        cfg,
        tool_result,
        conversations,
        model_mode: str = "general",
        original_prompt: Optional[str] = None,
        input_data_item: Dict = None,
    ):
        """
        处理工具结果，更新对话历史，生成新的提示
        
        参数:
            cfg: 工具配置
            tool_result: 工具调用返回的结果
            conversations: 当前对话历史
            model_mode (str): 生成更新提示的模式
            original_prompt (Optional[str]): 原始用户提示
            input_data_item (Dict): 包含图像和其他信息的输入数据项
            
        返回:
            更新后的对话历史
        """
        edited_image = None
        new_round_prompt = original_prompt
        image_key_info = ""
        item_id = id(input_data_item) if input_data_item else None
        
        # 获取工具名称
        api_name = cfg.get("API_name", cfg.get("api_name", "未知工具"))

        if tool_result is not None:
            try:
                # 如果工具结果中包含"edited_image"，则处理图像编辑
                if "edited_image" in tool_result:
                    try:
                        # 从结果中移除编辑后的图像并添加到历史
                        # 这时候edited_image已经被删除了
                        edited_image_base64 = tool_result.pop("edited_image")
                        if edited_image_base64 is not None:
                            # 将base64字符串转换为PIL图像
                            edited_image = base64_to_pil(edited_image_base64)
                            
                            # 添加到图像历史中
                            if item_id and item_id in self.image_history:
                                next_img_idx = len(self.image_history[item_id]) + 1
                                new_img_key = f"img_{next_img_idx}"
                                self.image_history[item_id][new_img_key] = edited_image
                                # 在工具结果中添加图像索引信息
                                image_key_info = f"\nNew image available as: {new_img_key}"
                                if isinstance(tool_result, dict):
                                    tool_result["image_key"] = new_img_key
                        else:
                            edited_image = None
                    except Exception as e:
                        edited_image = None
                

                # 将tool_result中的"tool_reward"删除
                tool_result.pop("tool_reward", None)
                
                # 从结果中获取API名称（支持多个键名）
                api_name = cfg.get("API_name", cfg.get("api_name", ""))

                new_response = f"OBSERVATION:\n{api_name} tool output: {tool_result}\n"
                new_round_prompt = (
                    f"{new_response}{image_key_info}\nPlease summarize the tool output content and answer my question."
                )

            except Exception as e:
                # 如果出现错误，恢复为原始提示
                edited_image = None
                new_round_prompt = original_prompt
        
        
        
        # 将新消息（包含文本和可选图像）追加到对话历史中
        updated_conversations = self.append_conversation_fn(
            conversation=conversations, 
            text=new_round_prompt, 
            image=edited_image if edited_image else None, 
            role="user"
        )

        return updated_conversations
        
    def get_repr_of_conversation(self, conversation):
        """
        获取对话的字符串表示
        
        参数:
            conversation: 对话列表
            
        返回:
            对话的字符串表示
        """
        conversation_str = ""
        for message in conversation:
            role = message["role"]
            # content = [item for item in message["content"] if isinstance(item, str) or item["type"] == "text"]
            content = message["content"]
            c_res = ""
            if isinstance(content, str):
                c_res = content
            elif isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type","") == "text":
                        c_res += str(c["text"])
            else:
                raise ValueError(f"Unknown content type: {type(content)}")
            conversation_str += f"{role}: {c_res}\n"
        return conversation_str.strip()
        
    def log_item_into_file(self, item, tool_log_file):
        """
        将对话项记录到文件中
        
        参数:
            item: 对话项
            tool_log_file: 日志文件路径
        """
        conversations = item["conversations"]
        conversations_str = self.get_repr_of_conversation(conversations)
        append_jsonl(conversations_str, tool_log_file)

    # def batch_inference(self, dataset):
    #     """
    #     批量推理函数，处理BaseEvalDataset数据集中的所有项目
        
    #     参数:
    #         dataset: 要处理的BaseEvalDataset数据集
    #     """
    #     # 创建数据加载器，批大小为1，工作线程数为2，使用collate_fn确保每次返回单个数据项
    #     dataloader = torch.utils.data.DataLoader(
    #         dataset, 
    #         batch_size=1, 
    #         num_workers=2, 
    #         collate_fn=lambda x: x[0]  # 确保每次返回一个数据项
    #     )
        
    #     # 创建进度条
    #     progress_bar = tqdm_rank0(len(dataloader), desc="Model Responding")
        
    #     # 处理数据集中的每个项目
    #     for idx, item in enumerate(dataloader):
    #         # 更新进度条
    #         progress_bar.update(1)
            
    #         # 准备输入数据
    #         input_data = {
    #             "conversation": item.get("conversation", None),
    #             "prompt": item.get("text", ""),
    #             "images": [item.get("image")] if item.get("image") is not None else []
    #         }
            
    #         if "system" in item:
    #             input_data["system"] = item["system"]
                
    #         # 执行推理
    #         results = self.inference(
    #             inputs=[input_data],
    #             max_rounds=self.max_rounds,
    #             do_sample=False
    #         )
            
    #         # 处理结果并存储到数据集中
    #         # 所以不需要return
    #         for result in results:
    #             idx = item["idx"]
    #             # 获取最后一个样本的结果
    #             sample = result["samples"][0]
    #             # 存储结果
    #             dataset.store_results(dict(idx=idx, results=sample))

    def extract_tool_call(self, text: str):
        """
        从模型响应文本中提取<tool_call>标签内的工具调用信息
        
        参数:
            text (str): 包含tool_call的模型响应文本
            
        返回:
            Optional[List[Dict]]: 解析后的工具调用列表，如果提取失败则返回None
        """
        try:
            # 使用正则表达式查找<tool_call>标签内的内容
            tool_call_pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
            tool_call_match = re.search(tool_call_pattern, text, re.DOTALL)
            
            if not tool_call_match:
                return None
                
            tool_call_content = tool_call_match.group(1).strip()
            
            # 尝试解析整个JSON数组
            try:
                # 首先尝试解析整个内容为JSON数组
                if tool_call_content.startswith('[') and tool_call_content.endswith(']'):
                    json_array = json.loads(tool_call_content)
                    if isinstance(json_array, list):
                        valid_objects = []
                        for obj in json_array:
                            if isinstance(obj, dict) and "name" in obj and "parameters" in obj:
                                valid_objects.append(obj)
                        if valid_objects:
                            return valid_objects
                
                # 如果不是JSON数组，尝试解析为单个JSON对象
                if (tool_call_content.startswith('{') and tool_call_content.endswith('}')):
                    json_obj = json.loads(tool_call_content)
                    if "name" in json_obj and "parameters" in json_obj:
                        return [json_obj]
            except json.JSONDecodeError:
                pass
            
            # 如果上述方法失败，尝试提取单个JSON对象
            json_objects = []
            # 使用正则表达式匹配所有JSON对象
            json_pattern = r'({[^{}]*(?:{[^{}]*}[^{}]*)*})'
            matches = re.finditer(json_pattern, tool_call_content, re.DOTALL)
            
            for match in matches:
                try:
                    json_obj = json.loads(match.group(1))
                    if isinstance(json_obj, dict) and "name" in json_obj and "parameters" in json_obj:
                        json_objects.append(json_obj)
                except json.JSONDecodeError:
                    continue
            
            if not json_objects:
                return None
                
            return json_objects
            
        except Exception as e:
            print(f"Error extracting tool call: {e}")
            return None

    def call_tool_with_retry(
        self, 
        api_name: str, 
        api_params: dict, 
        tool_retry_num: int, 
        tool_retry_interval: int, 
        tool_timeout_sec: int
    ) -> Dict:
        """
        调用工具并处理重试，作为并行处理的子任务
        
        参数:
            api_name: 工具API名称
            api_params: 工具参数
            tool_retry_num: 重试次数
            tool_retry_interval: 重试间隔
            tool_timeout_sec: 超时时间
            
        返回:
            工具调用结果
        """
        tool_result = {"text": f"Failed to call tool {api_name}", "error_code": 1}
        retry_info = []
        
        for attempt in range(tool_retry_num):
            try:
                # 设置超时警报
                signal.alarm(tool_timeout_sec)
                # 请求工具获得回答结果
                tool_result = self.tool_manager.call_tool(api_name, api_params)
                # 取消超时警报
                signal.alarm(0)
                break
            except TimeoutException as te:
                retry_info.append(f"第{attempt + 1}次尝试: 超时")
                tool_result = {"text": f"Failed to call tool {api_name}: {te}", "error_code": 1}
            except Exception as e:
                retry_info.append(f"第{attempt + 1}次尝试: 异常 - {str(e)}")
                tool_result = {"text": f"Failed to call tool {api_name}: {e}", "error_code": 1}
            finally:
                # 确保取消超时警报
                signal.alarm(0)
                if attempt < tool_retry_num - 1:
                    time.sleep(tool_retry_interval)
        
        return tool_result

    def process_tool_calls(
        self,
        input_data: List[Dict],
        input_idxs: List[int],
        tool_cfgs: List[Dict],
        tool_retry_num: int,
        tool_retry_interval: int,
        tool_timeout_sec: int
    ) -> List[Tuple[int, Dict, Dict]]:
        """
        并行处理多个工具调用
        
        参数:
            input_data: 输入数据列表
            input_idxs: 需要处理的输入索引列表
            tool_cfgs: 工具配置列表
            tool_retry_num: 重试次数
            tool_retry_interval: 重试间隔
            tool_timeout_sec: 超时时间
            
        返回:
            包含(索引, 工具配置, 工具结果)的元组列表
        """
        tool_tasks = []
        
        # 收集所有需要调用的工具任务
        for i, idx in enumerate(input_idxs):
            if tool_cfgs[i] is not None:
                try:
                    # 获取工具名称和参数
                    original_api_name = tool_cfgs[i][0].get("API_name")
                    api_params = tool_cfgs[i][0].get("API_params", {}).copy()
                    
                    # 工具名称映射表
                    tool_name_list = ['SegmentRegionAroundPoint', 'Point', 'OCR','DrawLine','Crop','GroundingDINO','LanguageModel','DrawLine','DrawShape','HighlightBox','MaskBox','GetSubplotInfo','GetBarInfo']
                    if original_api_name not in tool_name_list:
                        # 如果工具名称不在映射表中，则跳过
                        continue
                    api_name = original_api_name

                    item_id = id(input_data[idx])
                    
                    # 处理图像参数
                    if "image" in api_params:
                        img_key = api_params.get("image", "")
                        # 如果是img_n格式，则从图像历史中获取对应图像
                        if isinstance(img_key, str) and img_key.startswith("img_"):
                            if item_id in self.image_history and img_key in self.image_history[item_id]:
                                image = self.image_history[item_id][img_key]
                                # 转换为base64格式
                                image_base64 = pil_to_base64(image, url_format=False)
                                # 更新参数中的图像
                                api_params["image"] = image_base64
                            else:
                                # 如果找不到指定图像，设置为None
                                api_params["image"] = None
                        else:
                            # 如果不是img_n格式，也设置成None
                            api_params["image"] = None
                    
                    # 添加到任务列表
                    tool_tasks.append((idx, tool_cfgs[i], api_name, api_params))
                    
                except Exception as e:
                    # 处理错误，跳过这个工具调用
                    continue
        
        results = []
        
        # 使用线程池并行执行所有工具调用
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(16, len(tool_tasks))) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(
                    self.call_tool_with_retry, 
                    api_name, 
                    api_params, 
                    tool_retry_num, 
                    tool_retry_interval, 
                    tool_timeout_sec
                ): (idx, cfg) 
                for idx, cfg, api_name, api_params in tool_tasks
            }
            
            # 收集所有结果
            for future in concurrent.futures.as_completed(future_to_task):
                idx, cfg = future_to_task[future]
                try:
                    tool_result = future.result()
                    results.append((idx, cfg, tool_result))
                except Exception as e:
                    # 处理异常情况
                    tool_result = {"text": f"Failed to call tool: {str(e)}", "error_code": 1}
                    results.append((idx, cfg, tool_result))
        
        return results

    def inference(
        self,
        inputs: List[Dict],
        # 不是必须要设定的参数
        sampling_params: SamplingParams = None,
        max_rounds: int = 5,
        do_sample: bool = True,
        sample_num: int = 1,
        vllm_retry_num: int = 3,
        vllm_retry_interval: int = 5,
        vllm_timeout_sec: int = 500,
        tool_retry_num: int = 3,
        tool_retry_interval: int = 5,
        tool_timeout_sec: int = 60,
        tool_log_file: str = None,
    ) -> str:
        """
        执行模型推理和工具调用
        
        参数:
            inputs: 输入数据列表
            sampling_params: 采样参数
            max_rounds: 最大对话轮数
            do_sample: 是否进行采样
            sample_num: 采样数量
            vllm_retry_num: VLLM重试次数
            vllm_retry_interval: VLLM重试间隔时间(秒)
            vllm_timeout_sec: VLLM超时时间(秒)
            tool_retry_num: 工具调用重试次数
            tool_retry_interval: 工具调用重试间隔时间(秒)
            tool_timeout_sec: 工具调用超时时间(秒)
            tool_log_file: 工具日志文件路径
            
        返回:
            推理结果列表
        """
        # 重置工具调用统计
        self.tool_stats.reset()
        
        # 设置采样参数
        if not do_sample:
            kwargs = {
                "n": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "detokenize": True,
            }
        else:
            kwargs = {
                "n": 1,
                "detokenize": True,
            }
        new_sampling_params = deepcopy(sampling_params)
        for k,v in kwargs.items():
            setattr(new_sampling_params, k, v)
        
        # print(f"vllm_tool_inference.py的new_sampling_params: {new_sampling_params}")
        # print("vllm_tool_inferencer.py传入的inputs的维度为：", len(inputs))
            
        
        # 构建输入数据
        input_data = []
        for input_item_idx,item in enumerate(inputs):
            # 处理图像输入
            if "images" in item:
                images = item["images"]
                if not isinstance(images, list):
                    images = [images]
                    
            # 处理对话输入
            conversation = list(item["conversation"])
            first_content = conversation[0]["content"]
            new_first_content = []
            img_idx = 0
            # 处理第一条消息中的文本和图像
            for c in first_content:
                if c["type"] == "text":
                    new_first_content.append(c)
                elif c["type"] == "image":
                    image = images[img_idx]
                    image_base64 = pil_to_base64(image, url_format=True)
                    new_first_content.append({"type": "image_url", "image_url": {"url": image_base64}})
                    img_idx += 1
            prompt = new_first_content[-1]["text"]
            conversation[0]['role'] = 'user'
            conversation[0]["content"] = new_first_content[1:]
            initial_user_messages = conversation
            initial_user_messages.insert(0, {"role": "system", "content": new_first_content[0]['text']})
            # 如果do_sample了，那input_data就是num_roll_out*len(inputs)的数据
            if do_sample:
                for _ in range(sample_num):
                    data_instance = dict(
                        # 同一批采样数据的input_item_idx值相同
                        input_item_idx=input_item_idx,
                        conversations=initial_user_messages.copy(),
                        status="processing",
                        model_outputs=[],
                        model_output_ids=[],
                        tool_rewards=[],
                        tool_cfgs=[],
                        tool_outputs=[],
                        new_round_input=[],
                        prompt=prompt
                    )
                    input_data.append(data_instance)
                    # 初始化该数据实例的图像历史
                    item_id = id(data_instance) # 这里item_id是data_instance的id，是data_instance的内存地址
                    self.image_history[item_id] = {}
                    # 存储初始图像为img_1
                    if len(images) > 0:
                        self.image_history[item_id]["img_1"] = images[0]
                        # 添加其他图像（如果有）
                        for idx, img in enumerate(images[1:], start=2):
                            self.image_history[item_id][f"img_{idx}"] = img
            else:
                data_instance = dict(
                    input_item_idx=input_item_idx,
                    conversations=initial_user_messages.copy(),
                    status="processing",
                    model_outputs=[],
                    model_output_ids=[],
                    tool_rewards=[],
                    tool_cfgs=[],
                    tool_outputs=[],
                    new_round_input=[],
                    prompt=prompt
                )
                input_data.append(data_instance)
                # 初始化该数据实例的图像历史
                item_id = id(data_instance)
                self.image_history[item_id] = {}
                # 存储初始图像为img_1
                if len(images) > 0:
                    self.image_history[item_id]["img_1"] = images[0]
                    # 添加其他图像（如果有）
                    for idx, img in enumerate(images[1:], start=2):
                        self.image_history[item_id][f"img_{idx}"] = img
        # print("do_sample之后的input_data的维度为：", len(input_data))
        # print("构造好的image_history的长度为：", len(self.image_history))
            
        # 执行多轮对话
        for round_num in range(max_rounds):
            # 获取状态为"processing"的对话和索引
            # 一次性获取所有状态为"processing"的对话
            input_conversations = [item["conversations"] for item in input_data if item["status"] == "processing"]
            # input_idxs值从0开始，到len(input_data) - 1
            input_idxs = [idx for idx, item in enumerate(input_data) if item["status"] == "processing"]
            
            # 如果没有需要处理的对话，则退出循环
            if len(input_conversations) == 0:
                break
            
            # 模型推理，带重试机制
            outputs = None
            retry_info = []
            
            for attempt in range(vllm_retry_num):
                try:
                    # 设置超时警报
                    signal.alarm(vllm_timeout_sec)
                    # 获得所有状态为"processing"的对话的输出，批量生成的
                    outputs = self.vllm_model.chat(
                        input_conversations, sampling_params=new_sampling_params, use_tqdm = True,
                    )
                    # 取消超时警报
                    signal.alarm(0)
                    break
                except TimeoutException as te:
                    retry_info.append(f"第{attempt + 1}次尝试: 超时")
                except Exception as e:
                    retry_info.append(f"第{attempt + 1}次尝试: 异常 - {str(e)}")
                finally:
                    # 确保取消超时警报
                    signal.alarm(0)
                    if attempt < vllm_retry_num - 1:
                        time.sleep(vllm_retry_interval)
            
            # 如果有重试信息，打印出来
            if retry_info:
                print("\n========== VLLM模型推理重试 ==========")
                for info in retry_info:
                    print(info)
                print("================================\n")
                
            # 处理模型输出
            if outputs is None:
                # 如果模型生成失败，使用错误消息
                output_texts = ["Model generation error"] * len(input_conversations)
                output_idss = [(1712, 9471, 1465, 151645)] * len(input_conversations)
            else:
                output_texts = [output.outputs[0].text for output in outputs]
                output_idss = [output.outputs[0].token_ids for output in outputs]

            # 首先处理所有模型输出，提取工具调用信息
            tool_cfgs_list = []
            for input_idx, output_text, output_ids in zip(input_idxs, output_texts, output_idss):
                # 记录模型输出
                input_data[input_idx]["model_outputs"].append(output_text)
                input_data[input_idx]["model_output_ids"].append(output_ids)
                # 将模型回复添加到对话中
                input_data[input_idx]["conversations"] = self.append_conversation_fn(conversation=input_data[input_idx]["conversations"], text=output_text, role="assistant")
                
                # 如果回复中包含"<response>"，则标记为已完成并继续处理下一个
                if "<response>" in output_text:
                    input_data[input_idx]["status"] = "finished"
                    input_data[input_idx]["tool_rewards"].append(0.0)
                    # 立即清理该对话的图像历史
                    item_id = id(input_data[input_idx])
                    if item_id in self.image_history:
                        del self.image_history[item_id]
                    tool_cfgs_list.append(None)
                    continue

                # 提取工具调用信息
                tool_calls = self.extract_tool_call(output_text)
                if tool_calls is not None and len(tool_calls) > 0:
                    # 只使用第一个工具调用
                    tool_call = tool_calls[0]
                    tool_name = tool_call['name']
                    tool_params = tool_call['parameters']
                    
                    # 构建工具配置格式
                    tool_cfg = [{
                        "API_name": tool_name,
                        "API_params": tool_params
                    }]
                    tool_cfgs_list.append(tool_cfg)
                else:
                    tool_cfg = None
                    tool_cfgs_list.append(None)
                    # 如果模型输出中没有工具调用，则将工具调用奖励设置为0
                    input_data[input_idx]["tool_rewards"].append(0.0)
                    
                    # 如果没有工具配置，则将原始提示添加到对话中
                    input_data[input_idx]["conversations"] = self.append_conversation_fn(
                        conversation=input_data[input_idx]["conversations"], 
                        text=input_data[input_idx]["prompt"], 
                        role="user"
                    )
            
            # 并行处理所有工具调用
            tool_results = self.process_tool_calls(
                input_data=input_data,
                input_idxs=input_idxs,
                tool_cfgs=tool_cfgs_list,
                tool_retry_num=tool_retry_num,
                tool_retry_interval=tool_retry_interval,
                tool_timeout_sec=tool_timeout_sec
            )
            
            # 处理工具调用结果
            for idx, tool_cfg, tool_result in tool_results:
                # 记录工具配置和输出
                input_data[idx]["tool_cfgs"].append(tool_cfg)
                input_data[idx]["tool_outputs"].append(tool_result)
                
                # 获取API名称
                api_name = tool_cfg[0].get("API_name", "未知工具")
                
                # 获取调用状态
                status = tool_result.get("status", "")
                error_code = tool_result.get("error_code", -1)
                # 将tool_reward添加到工具调用奖励列表中
                if tool_result.get("tool_reward") is not None:
                    input_data[idx]["tool_rewards"].append(tool_result.get("tool_reward"))
                call_status = "success" if (status == "success" or error_code == 0) else "failed"
                
                # 更新工具调用统计
                self.tool_stats.add_call(call_status == "success", api_name)
                
                # 处理工具结果并更新对话
                updated_conversations = self.handle_tool_result(
                    cfg=tool_cfg[0],
                    tool_result=tool_result,
                    conversations=input_data[idx]["conversations"],
                    model_mode="general",
                    original_prompt=input_data[idx]["prompt"],
                    input_data_item=input_data[idx]
                )
                
                # 更新对话
                input_data[idx]["conversations"] = updated_conversations
                
        # 记录工具输出到日志文件
        if tool_log_file is not None:
            for item in input_data:
                self.log_item_into_file(item, tool_log_file)
                
        # 收集最终结果
        results = []
        # 按输入项索引分组结果
        grouped_results = {}
        for item in input_data:
            input_item_idx = item["input_item_idx"]
            if grouped_results.get(input_item_idx) is None:
                grouped_results[input_item_idx] = []
            grouped_results[input_item_idx].append(item)
            # 保存对话到日志文件
            log_file = self.save_conversation_to_log(item["conversations"])
            print(f"对话已完成并保存到: {log_file}")

            # 只清理未完成的对话的图像历史（已完成的在前面已经清理）
            item_id = id(item)
            if item_id in self.image_history:
                del self.image_history[item_id]
        
        # 将分组结果转换为最终输出格式
        for k,v in grouped_results.items():
            results.append({
                "item_idx": k, 
                "samples": v,
                "tool_stats": self.tool_stats.get_stats_dict()  # 添加工具调用统计
            })
        # print("vllm_tool_inferencer.py的results的维度为：", len(results))
        # print("vllm_tool_inferencer.py运行结束后的self.tool_stats.get_stats_dict()：", self.tool_stats.get_stats_dict())
        return results

    def save_conversation_to_log(self, conversation, log_dir="/mnt/petrelfs/sunhaoyu/visual-code/EasyR1/scripts/logs"):
        """
        将完成的对话保存到单个日志文件中，图像使用占位符替换
        使用追加方式写入，确保不会覆盖已有内容
        
        参数:
            conversation: 对话内容
            log_dir: 日志目录
        """
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        # 使用固定的日志文件名，按日期分类
        # today = datetime.datetime.now().strftime("%Y%m%d")
        # log_file = os.path.join(log_dir, f"conversations_{today}.jsonl")
        # log_file = os.path.join(log_dir, f"conversations_spot.jsonl")
        log_file = os.path.join(log_dir, f"conversations_reserved.jsonl")
        
        # 处理对话内容，替换图像为占位符
        processed_conversation = []
        image_counter = 1
        
        for message in conversation:
            processed_message = {"role": message["role"]}
            
            if isinstance(message["content"], list):
                processed_content = []
                for item in message["content"]:
                    if isinstance(item, dict):
                        if item.get("type") == "image_url":
                            # 替换图像为占位符
                            processed_content.append({
                                "type": "image_url",
                                "image_url": {"url": f"<image{image_counter}>"}
                            })
                            image_counter += 1
                        else:
                            # 保留其他类型的内容
                            processed_content.append(item)
                    else:
                        processed_content.append(item)
                processed_message["content"] = processed_content
            else:
                processed_message["content"] = message["content"]
            
            processed_conversation.append(processed_message)
        
        # 添加时间戳
        conversation_entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "conversation": processed_conversation
        }
        
        # 使用文件锁确保线程安全，以追加模式写入单个对话条目
        with _get_file_lock(log_file):
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(conversation_entry, ensure_ascii=False) + '\n')
        
        return log_file

