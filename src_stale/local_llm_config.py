import os
import json
from typing import Dict, Any, Optional
import requests
from dotenv import load_dotenv

load_dotenv()


class LocalLLMConfig:
    """本地LLM模型配置和管理"""
    
    def __init__(self):
        self.config = {
            "ollama": {
                "base_url": "http://localhost:11434",
                "models": {
                    "qwen2.5": {
                        "model_name": "qwen2.5:latest", 
                        "context_length": 32768,
                        "description": "Qwen2.5模型，适合中文金融分析"
                    },
                    "deepseek-coder": {
                        "model_name": "deepseek-coder:latest",
                        "context_length": 16384,
                        "description": "DeepSeek-Coder模型，适合代码和逻辑分析"
                    }
                }
            },
            "lm_studio": {
                "base_url": "http://localhost:1234",
                "description": "LM Studio本地API服务"
            },
            "text_generation_webui": {
                "base_url": "http://localhost:5000",
                "description": "Text Generation WebUI API"
            }
        }
    
    def check_ollama_status(self) -> bool:
        """检查Ollama服务状态"""
        try:
            response = requests.get(f"{self.config['ollama']['base_url']}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_available_models(self) -> Dict[str, Any]:
        """列出可用的模型"""
        available_models = {}
        
        # 检查Ollama
        if self.check_ollama_status():
            try:
                response = requests.get(f"{self.config['ollama']['base_url']}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    available_models["ollama"] = {
                        "status": "available",
                        "models": [model["name"] for model in data.get("models", [])]
                    }
                else:
                    available_models["ollama"] = {"status": "error", "models": []}
            except Exception as e:
                available_models["ollama"] = {"status": "error", "error": str(e)}
        else:
            available_models["ollama"] = {"status": "unavailable"}
        
        return available_models
    
    def get_model_suggestions(self) -> Dict[str, str]:
        """获取模型安装建议"""
        return {
            "qwen2.5": "ollama pull qwen2.5:latest",
            "qwen2.5-14b": "ollama pull qwen2.5:14b", 
            "deepseek-coder": "ollama pull deepseek-coder:latest",
            "llama3.1": "ollama pull llama3.1:latest",
            "mistral": "ollama pull mistral:latest"
        }
    
    def generate_ollama_chat_payload(self, model: str, prompt: str, system_prompt: Optional[str] = None, messages: Optional[list] = None) -> Dict[str, Any]:
        """生成Ollama Chat API请求载荷"""
        # 构建消息列表
        chat_messages = []
        
        # 如果有历史消息，先添加历史消息
        if messages:
            chat_messages.extend(messages)
        
        # 添加当前用户消息
        chat_messages.append({
            "role": "user",
            "content": prompt
        })
        
        payload = {
            "model": model,
            "messages": chat_messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 2000  # chat API 使用 num_predict 而不是 max_tokens
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        return payload
    
    def call_ollama_model(self, model: str, prompt: str, system_prompt: Optional[str] = None, messages: Optional[list] = None) -> str:
        """调用Ollama模型（使用Chat API）"""
        if not self.check_ollama_status():
            return "错误: Ollama服务不可用，请确保Ollama已启动"
        
        payload = self.generate_ollama_chat_payload(model, prompt, system_prompt, messages)
        
        try:
            response = requests.post(
                f"{self.config['ollama']['base_url']}/api/chat",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                # chat API 返回格式: {"message": {"role": "assistant", "content": "..."}}
                message = result.get("message", {})
                return message.get("content", "无响应内容")
            else:
                return f"错误: HTTP {response.status_code}"
                
        except Exception as e:
            return f"调用模型时出错: {str(e)}"
    
    def setup_instructions(self) -> str:
        """返回设置说明"""
        return """
# 本地LLM模型设置指南

## 1. 安装Ollama (推荐)
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# 下载并安装: https://ollama.ai/download
```

## 2. 启动Ollama服务
```bash
ollama serve
```

## 3. 下载推荐模型
```bash
# 中文金融分析推荐
ollama pull qwen2.5:14b

# 英文通用推荐  
ollama pull llama3.1:8b

# 代码分析推荐
ollama pull deepseek-coder:6.7b
```

## 4. 验证安装
```bash
ollama list
```

## 替代方案

### LM Studio (GUI界面)
1. 下载: https://lmstudio.ai/
2. 启动并下载模型
3. 启用本地服务器

### Text Generation WebUI
1. 安装: https://github.com/oobabooga/text-generation-webui
2. 启动WebUI
3. 启用API模式

## 模型推荐

### 金融分析场景:
- **Qwen2.5-14B**: 中文金融理解最佳
- **Qwen2.5-7B**: 平衡性能和资源
- **Qwen2.5-32B**: 高端分析（需要更多显存）

### 硬件要求:
- RTX 4090 (24GB): 可运行14B模型
- RTX 4080 (16GB): 适合7B-8B模型  
- RTX 4070 (12GB): 适合7B模型
"""


def main():
    """主函数：检查和配置本地LLM"""
    config = LocalLLMConfig()
    
    print("=== 本地LLM配置检查 ===")
    
    # 检查可用模型
    available = config.list_available_models()
    print(f"可用服务: {json.dumps(available, indent=2, ensure_ascii=False)}")
    
    # 显示安装建议
    if not available.get("ollama", {}).get("status") == "available":
        print("\n=== 设置指南 ===")
        print(config.setup_instructions())
    else:
        print("\n=== 模型安装建议 ===")
        suggestions = config.get_model_suggestions()
        for model, command in suggestions.items():
            print(f"{model}: {command}")
    
    # 如果有可用模型，测试调用
    ollama_status = available.get("ollama", {})
    if (ollama_status.get("status") == "available" and 
        ollama_status.get("models")):
        
        print(f"\n=== 测试模型调用 ===")
        models = ollama_status["models"]
        test_model = models[0]
        
        print(f"使用模型: {test_model}")
        
        test_prompt = "请简单介绍一下中国股票市场的主要特点。"
        system_prompt = "你是一位专业的金融分析师，请用简洁专业的语言回答问题。"
        
        response = config.call_ollama_model(test_model, test_prompt, system_prompt)
        print(f"回答: {response}")


if __name__ == "__main__":
    main()