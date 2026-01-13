"""
函数式编程版本的本地 LLM 配置模块
使用纯函数、函数组合和不可变数据结构
"""

import json
import requests
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from functools import partial
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# 数据类型定义 - 使用不可变的 dataclass
# ============================================================================

@dataclass(frozen=True)
class ServiceConfig:
    """不可变的服务配置"""
    base_url: str
    description: str


@dataclass(frozen=True)
class ModelConfig:
    """不可变的模型配置"""
    model_name: str
    context_length: int
    description: str


@dataclass(frozen=True)
class OllamaConfig:
    """不可变的 Ollama 配置"""
    base_url: str
    models: Dict[str, ModelConfig]


@dataclass(frozen=True)
class SystemConfig:
    """不可变的系统配置"""
    ollama: OllamaConfig
    lm_studio: ServiceConfig
    text_generation_webui: ServiceConfig


@dataclass(frozen=True)
class ServiceStatus:
    """不可变的服务状态"""
    is_available: bool
    models: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass(frozen=True)
class ChatPayload:
    """不可变的聊天载荷"""
    model: str
    messages: List[Dict[str, str]]
    stream: bool
    options: Dict[str, Any]
    system: Optional[str] = None


# ============================================================================
# 配置创建函数 - 纯函数
# ============================================================================

def create_model_config(
    model_name: str,
    context_length: int,
    description: str
) -> ModelConfig:
    """创建模型配置"""
    return ModelConfig(
        model_name=model_name,
        context_length=context_length,
        description=description
    )


def create_default_models() -> Dict[str, ModelConfig]:
    """创建默认模型配置字典 - 纯函数"""
    return {
        "qwen2.5": create_model_config(
            model_name="qwen2.5:latest",
            context_length=32768,
            description="Qwen2.5模型，适合中文金融分析"
        ),
        "deepseek-coder": create_model_config(
            model_name="deepseek-coder:latest",
            context_length=16384,
            description="DeepSeek-Coder模型，适合代码和逻辑分析"
        )
    }


def create_ollama_config(
    base_url: str = "http://localhost:11434",
    models: Optional[Dict[str, ModelConfig]] = None
) -> OllamaConfig:
    """创建 Ollama 配置 - 纯函数"""
    if models is None:
        models = create_default_models()

    return OllamaConfig(base_url=base_url, models=models)


def create_service_config(base_url: str, description: str) -> ServiceConfig:
    """创建服务配置 - 纯函数"""
    return ServiceConfig(base_url=base_url, description=description)


def create_system_config(
    ollama_base_url: str = "http://localhost:11434",
    lm_studio_base_url: str = "http://localhost:1234",
    webui_base_url: str = "http://localhost:5000"
) -> SystemConfig:
    """创建系统配置 - 纯函数"""
    return SystemConfig(
        ollama=create_ollama_config(ollama_base_url),
        lm_studio=create_service_config(
            lm_studio_base_url,
            "LM Studio本地API服务"
        ),
        text_generation_webui=create_service_config(
            webui_base_url,
            "Text Generation WebUI API"
        )
    )


# ============================================================================
# 纯函数 - 数据转换
# ============================================================================

def build_chat_messages(
    prompt: str,
    system_prompt: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    """构建聊天消息列表 - 纯函数"""
    messages = []

    # 添加历史消息
    if history:
        messages.extend(history)

    # 添加当前用户消息
    messages.append({
        "role": "user",
        "content": prompt
    })

    return messages


def create_chat_payload(
    model: str,
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_predict: int = 2000,
    stream: bool = False
) -> ChatPayload:
    """创建聊天请求载荷 - 纯函数"""
    options = {
        "temperature": temperature,
        "top_p": top_p,
        "num_predict": num_predict
    }

    return ChatPayload(
        model=model,
        messages=messages,
        stream=stream,
        options=options,
        system=system_prompt
    )


def payload_to_dict(payload: ChatPayload) -> Dict[str, Any]:
    """将载荷转换为字典 - 纯函数"""
    result = {
        "model": payload.model,
        "messages": payload.messages,
        "stream": payload.stream,
        "options": payload.options
    }

    if payload.system:
        result["system"] = payload.system

    return result


def extract_models_from_response(response_data: Dict[str, Any]) -> List[str]:
    """从响应中提取模型列表 - 纯函数"""
    models = response_data.get("models", [])
    return [model["name"] for model in models]


def extract_content_from_chat_response(response_data: Dict[str, Any]) -> str:
    """从聊天响应中提取内容 - 纯函数"""
    message = response_data.get("message", {})
    return message.get("content", "无响应内容")


# ============================================================================
# 高阶函数 - 函数组合
# ============================================================================

def with_error_handling(default_result: Any) -> Callable:
    """高阶函数：为函数添加错误处理"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if isinstance(default_result, ServiceStatus):
                    return ServiceStatus(
                        is_available=False,
                        models=[],
                        error=str(e)
                    )
                return default_result
        return wrapper
    return decorator


def with_timeout(timeout: int = 5) -> Callable:
    """高阶函数：为 HTTP 请求添加超时"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            kwargs['timeout'] = kwargs.get('timeout', timeout)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# 副作用函数 - HTTP 请求
# ============================================================================

@with_timeout(5)
def _http_get(url: str, **kwargs) -> requests.Response:
    """HTTP GET 请求（副作用）"""
    return requests.get(url, **kwargs)


@with_timeout(60)
def _http_post(url: str, json_data: Dict[str, Any], **kwargs) -> requests.Response:
    """HTTP POST 请求（副作用）"""
    return requests.post(url, json=json_data, **kwargs)


@with_error_handling(ServiceStatus(is_available=False))
def check_service_status(base_url: str) -> ServiceStatus:
    """检查服务状态（副作用）"""
    response = _http_get(f"{base_url}/api/tags")

    if response.status_code == 200:
        data = response.json()
        models = extract_models_from_response(data)
        return ServiceStatus(is_available=True, models=models)
    else:
        return ServiceStatus(
            is_available=False,
            error=f"HTTP {response.status_code}"
        )


@with_error_handling("")
def call_chat_api(base_url: str, payload: ChatPayload) -> str:
    """调用聊天 API（副作用）"""
    payload_dict = payload_to_dict(payload)
    response = _http_post(f"{base_url}/api/chat", payload_dict)

    if response.status_code == 200:
        data = response.json()
        return extract_content_from_chat_response(data)
    else:
        return f"错误: HTTP {response.status_code}"


# ============================================================================
# 公共 API - 组合纯函数和副作用函数
# ============================================================================

def check_ollama_status(config: SystemConfig) -> ServiceStatus:
    """
    检查 Ollama 服务状态

    Args:
        config: 系统配置

    Returns:
        ServiceStatus: 服务状态
    """
    return check_service_status(config.ollama.base_url)


def list_available_models(config: SystemConfig) -> Dict[str, ServiceStatus]:
    """
    列出所有服务的可用模型

    Args:
        config: 系统配置

    Returns:
        Dict[str, ServiceStatus]: 服务状态映射
    """
    return {
        "ollama": check_service_status(config.ollama.base_url)
    }


def get_model_suggestions() -> Dict[str, str]:
    """
    获取模型安装建议 - 纯函数

    Returns:
        Dict[str, str]: 模型名称到安装命令的映射
    """
    return {
        "qwen2.5": "ollama pull qwen2.5:latest",
        "qwen2.5-14b": "ollama pull qwen2.5:14b",
        "deepseek-coder": "ollama pull deepseek-coder:latest",
        "llama3.1": "ollama pull llama3.1:latest",
        "mistral": "ollama pull mistral:latest"
    }


def chat(
    config: SystemConfig,
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.7
) -> str:
    """
    调用 Ollama 模型进行聊天

    Args:
        config: 系统配置
        model: 模型名称
        prompt: 用户提示
        system_prompt: 系统提示（可选）
        history: 对话历史（可选）
        temperature: 温度参数

    Returns:
        str: 模型响应
    """
    # 检查服务状态
    status = check_ollama_status(config)
    if not status.is_available:
        return f"错误: Ollama服务不可用 - {status.error or '未知错误'}"

    # 构建消息（纯函数）
    messages = build_chat_messages(prompt, system_prompt, history)

    # 创建载荷（纯函数）
    payload = create_chat_payload(
        model=model,
        messages=messages,
        system_prompt=system_prompt,
        temperature=temperature
    )

    # 调用 API（副作用）
    return call_chat_api(config.ollama.base_url, payload)


def get_setup_instructions() -> str:
    """
    获取设置说明 - 纯函数

    Returns:
        str: 设置说明文本
    """
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


# ============================================================================
# 便捷函数 - 使用偏函数应用
# ============================================================================

def create_chat_function(config: SystemConfig, model: str) -> Callable:
    """
    创建一个预配置的聊天函数
    使用偏函数应用固定配置和模型
    """
    return partial(chat, config=config, model=model)


# ============================================================================
# 使用示例
# ============================================================================

def main():
    """主函数：检查和配置本地LLM"""
    # 创建配置
    config = create_system_config()

    print("=== 本地LLM配置检查 ===")

    # 检查可用模型
    available = list_available_models(config)
    print(f"可用服务: {json.dumps({k: {'available': v.is_available, 'models': v.models} for k, v in available.items()}, indent=2, ensure_ascii=False)}")

    # 获取 Ollama 状态
    ollama_status = available.get("ollama")

    # 显示安装建议
    if not ollama_status or not ollama_status.is_available:
        print("\n=== 设置指南 ===")
        print(get_setup_instructions())
    else:
        print("\n=== 模型安装建议 ===")
        suggestions = get_model_suggestions()
        for model, command in suggestions.items():
            print(f"{model}: {command}")

    # 如果有可用模型，测试调用
    if ollama_status and ollama_status.is_available and ollama_status.models:
        print(f"\n=== 测试模型调用 ===")
        test_model = ollama_status.models[0]

        print(f"使用模型: {test_model}")

        test_prompt = "请简单介绍一下中国股票市场的主要特点。"
        system_prompt = "你是一位专业的金融分析师，请用简洁专业的语言回答问题。"

        response = chat(
            config=config,
            model=test_model,
            prompt=test_prompt,
            system_prompt=system_prompt
        )
        print(f"回答: {response}")


if __name__ == "__main__":
    main()
