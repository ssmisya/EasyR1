import wandb
import sys
import requests
import json

def test_wandb_connection():
    try:
        # 尝试初始化一个wandb运行实例
        wandb.init(project="test_connection", name="test_run")
        
        # 记录一些简单的指标
        for i in range(10):
            wandb.log({"metric": i})
        
        # 完成运行
        wandb.finish()
        
        print("成功连接到wandb！")
        return True
    except Exception as e:
        print(f"连接wandb失败: {e}")
        return False

def get_models_list():
    try:
        url = "http://SH-IDC1-10-140-37-6:21112/list_models"
        # 完全禁用所有代理设置
        session = requests.Session()
        session.trust_env = False  # 不使用环境变量中的代理设置
        response = session.post(url, proxies={})
        
        # 检查请求是否成功
        response.raise_for_status()
        
        # 尝试解析JSON响应
        try:
            data = response.json()
            print("获取模型列表成功:")
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return data
        except ValueError:
            # 如果响应不是JSON格式
            print("响应不是JSON格式:")
            print(response.text)
            return response.text
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None

if __name__ == "__main__":
    # 获取模型列表
    models = get_models_list()
    
    # 测试wandb连接
    success = test_wandb_connection()
    sys.exit(0 if success else 1)