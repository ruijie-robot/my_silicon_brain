"""
函数式编程版本的 Ollama LLM 接口
使用纯函数、函数组合和不可变数据结构
"""

import ollama
from typing import Optional, List, Dict, Any, Callable, Iterator
from dataclasses import dataclass
from functools import partial, reduce
from ollama import chat, embeddings
from ollama import ChatResponse


# ============================================================================
# 数据类型定义 - 使用不可变的 dataclass
# ============================================================================

@dataclass(frozen=True)
class LLMConfig:
    """不可变的 LLM 配置"""
    model: str
    embed_model: str
    temperature: float = 0.8
    num_predict: int = 2000


@dataclass(frozen=True)
class Message:
    """不可变的消息"""
    role: str
    content: str


@dataclass(frozen=True)
class ChatResult:
    """不可变的聊天结果"""
    content: str
    error: Optional[str] = None

    @property
    def is_success(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class EmbeddingResult:
    """不可变的嵌入结果"""
    embedding: List[float]
    error: Optional[str] = None

    @property
    def is_success(self) -> bool:
        return self.error is None


# ============================================================================
# 纯函数 - 无副作用的数据转换
# ============================================================================

def create_config(
    model: str = 'qwen3:0.6b',
    embed_model: str = 'qwen3-embedding:0.6b',
    temperature: float = 0.8,
    num_predict: int = 2000
) -> LLMConfig:
    """创建不可变的配置对象"""
    return LLMConfig(
        model=model,
        embed_model=embed_model,
        temperature=temperature,
        num_predict=num_predict
    )


def create_message(role: str, content: str) -> Message:
    """创建不可变的消息对象"""
    return Message(role=role, content=content)


def create_system_message(content: str) -> Message:
    """创建系统消息的便捷函数"""
    return create_message("system", content)


def create_user_message(content: str) -> Message:
    """创建用户消息的便捷函数"""
    return create_message("user", content)


def create_assistant_message(content: str) -> Message:
    """创建助手消息的便捷函数"""
    return create_message("assistant", content)


def message_to_dict(message: Message) -> Dict[str, str]:
    """将消息转换为字典格式"""
    return {"role": message.role, "content": message.content}


def messages_to_dicts(messages: List[Message]) -> List[Dict[str, str]]:
    """将消息列表转换为字典列表"""
    return list(map(message_to_dict, messages))


def build_messages(
    prompt: str,
    system_prompt: Optional[str] = None,
    history: Optional[List[Message]] = None
) -> List[Message]:
    """构建消息列表 - 纯函数"""
    messages = []

    # 添加系统消息
    if system_prompt:
        messages.append(create_system_message(system_prompt))

    # 添加历史消息
    if history:
        messages.extend(history)

    # 添加当前用户消息
    messages.append(create_user_message(prompt))

    return messages


def extract_content_from_response(response: Dict[str, Any]) -> str:
    """从响应中提取内容 - 纯函数"""
    return response.get('message', {}).get('content', '')


def extract_embedding_from_response(response: Dict[str, Any]) -> List[float]:
    """从响应中提取嵌入向量 - 纯函数"""
    embeddings = response.get('embeddings', [])
    return embeddings[0] if embeddings else []


# ============================================================================
# 高阶函数 - 函数组合和错误处理
# ============================================================================

def with_error_handling(func: Callable) -> Callable:
    """高阶函数：为函数添加错误处理装饰器"""
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            # 根据返回类型返回相应的错误结果
            if 'embed' in func.__name__:
                return EmbeddingResult(embedding=[], error=str(e))
            else:
                return ChatResult(content="", error=str(e))
    return wrapper


def compose(*functions: Callable) -> Callable:
    """函数组合：从右到左组合函数"""
    def inner(arg):
        return reduce(lambda acc, func: func(acc), reversed(functions), arg)
    return inner


def pipe(*functions: Callable) -> Callable:
    """函数管道：从左到右应用函数"""
    def inner(arg):
        return reduce(lambda acc, func: func(acc), functions, arg)
    return inner


# ============================================================================
# 副作用函数 - 与外部系统交互（明确标识）
# ============================================================================

def _call_ollama_client(client: ollama.Client) -> Callable:
    """返回一个与 Ollama 客户端交互的函数（柯里化）"""
    def call_chat(config: LLMConfig, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """调用 Ollama 聊天 API（副作用）"""
        return ollama.chat(
            model=config.model,
            messages=messages,
            options={
                'temperature': config.temperature,
                'num_predict': config.num_predict
            }
        )
    return call_chat


def _call_ollama_embed(client: ollama.Client) -> Callable:
    """返回一个与 Ollama 嵌入 API 交互的函数（柯里化）"""
    def call_embed(config: LLMConfig, text: str) -> Dict[str, Any]:
        """调用 Ollama 嵌入 API（副作用）"""
        return client.embed(
            model=config.embed_model,
            input=text
        )
    return call_embed


def _call_ollama_stream(client: ollama.Client) -> Callable:
    """返回一个与 Ollama 流式 API 交互的函数（柯里化）"""
    def call_stream(config: LLMConfig, messages: List[Dict[str, str]]) -> Iterator[Dict[str, Any]]:
        """调用 Ollama 流式聊天 API（副作用）"""
        return ollama.chat(
            model=config.model,
            messages=messages,
            stream=True,
        )
    return call_stream


def _list_ollama_models(client: ollama.Client) -> List[str]:
    """列出可用模型（副作用）"""
    try:
        models = client.list()
        return [model.model for model in models['models']]
    except Exception as e:
        print(f"获取模型列表失败: {e}")
        return []


# ============================================================================
# 公共 API - 组合纯函数和副作用函数
# ============================================================================

@with_error_handling
def simple_chat(
    config: LLMConfig,
    prompt: str,
    system_prompt: Optional[str] = None,
    client: Optional[ollama.Client] = None
) -> ChatResult:
    """
    简单聊天接口 - 函数式 API

    Args:
        config: LLM 配置
        prompt: 用户提示
        system_prompt: 系统提示（可选）
        client: Ollama 客户端（可选，默认创建新客户端）

    Returns:
        ChatResult: 包含响应内容或错误信息
    """
    # 使用默认客户端
    if client is None:
        client = ollama.Client()

    # 构建消息（纯函数）
    messages = build_messages(prompt, system_prompt)
    messages_dict = messages_to_dicts(messages)

    # 调用 API（副作用）
    call_chat = _call_ollama_client(client)
    response = call_chat(config, messages_dict)

    # 提取内容（纯函数）
    content = extract_content_from_response(response)

    return ChatResult(content=content)


@with_error_handling
def chat_with_history(
    config: LLMConfig,
    messages: List[Message],
    client: Optional[ollama.Client] = None
) -> ChatResult:
    """
    带历史记录的聊天接口 - 函数式 API

    Args:
        config: LLM 配置
        messages: 消息历史
        client: Ollama 客户端（可选）

    Returns:
        ChatResult: 包含响应内容或错误信息
    """
    # 使用默认客户端
    if client is None:
        client = ollama.Client()

    # 转换消息格式（纯函数）
    messages_dict = messages_to_dicts(messages)

    # 调用 API（副作用）
    call_chat = _call_ollama_client(client)
    response = call_chat(config, messages_dict)

    # 提取内容（纯函数）
    content = extract_content_from_response(response)

    return ChatResult(content=content)


def stream_chat(
    config: LLMConfig,
    prompt: str,
    system_prompt: Optional[str] = None,
    client: Optional[ollama.Client] = None
) -> Iterator[str]:
    """
    流式聊天接口 - 函数式 API

    Args:
        config: LLM 配置
        prompt: 用户提示
        system_prompt: 系统提示（可选）
        client: Ollama 客户端（可选）

    Yields:
        str: 流式输出的内容片段
    """
    # 使用默认客户端
    if client is None:
        client = ollama.Client()

    try:
        # 构建消息（纯函数）
        messages = build_messages(prompt, system_prompt)
        messages_dict = messages_to_dicts(messages)

        # 调用流式 API（副作用）
        call_stream = _call_ollama_stream(client)
        stream = call_stream(config, messages_dict)

        # 流式输出
        for chunk in stream:
            content = chunk.get('message', {}).get('content', '')
            if content:
                yield content

    except Exception as e:
        yield f"错误: {str(e)}"


@with_error_handling
def embed_text(
    config: LLMConfig,
    text: str,
    client: Optional[ollama.Client] = None
) -> EmbeddingResult:
    """
    文本嵌入接口 - 函数式 API

    Args:
        config: LLM 配置
        text: 要嵌入的文本
        client: Ollama 客户端（可选）

    Returns:
        EmbeddingResult: 包含嵌入向量或错误信息
    """
    # 使用默认客户端
    if client is None:
        client = ollama.Client()

    # 调用嵌入 API（副作用）
    call_embed = _call_ollama_embed(client)
    response = call_embed(config, text)

    # 提取嵌入向量（纯函数）
    embedding = extract_embedding_from_response(response)

    return EmbeddingResult(embedding=embedding)


def list_models(client: Optional[ollama.Client] = None) -> List[str]:
    """
    列出可用模型 - 函数式 API

    Args:
        client: Ollama 客户端（可选）

    Returns:
        List[str]: 可用模型列表
    """
    if client is None:
        client = ollama.Client()

    return _list_ollama_models(client)


# ============================================================================
# 便捷函数 - 使用部分应用（Partial Application）
# ============================================================================

def create_chat_function(config: LLMConfig, client: Optional[ollama.Client] = None) -> Callable:
    """
    创建一个预配置的聊天函数
    使用偏函数应用固定配置
    """
    return partial(simple_chat, config=config, client=client)


def create_embed_function(config: LLMConfig, client: Optional[ollama.Client] = None) -> Callable:
    """
    创建一个预配置的嵌入函数
    使用偏函数应用固定配置
    """
    return partial(embed_text, config=config, client=client)


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 创建配置
    config = create_config(
        model='qwen3:0.6b',
        embed_model='qwen3-embedding:0.6b'
    )

    # 创建客户端
    client = ollama.Client()

    # 列出模型
    models = list_models(client)
    print(f"可用模型: {models}")

    # 简单聊天
    result = simple_chat(
        config=config,
        prompt="简述中国股市特点",
        system_prompt="你是专业的金融分析师",
        client=client
    )

    if result.is_success:
        print(f"回答: {result.content}")
    else:
        print(f"错误: {result.error}")

    # 嵌入示例
    embed_result = embed_text(
        config=config,
        text='The quick brown fox jumps over the lazy dog.',
        client=client
    )

    if embed_result.is_success:
        print(f"嵌入向量维度: {len(embed_result.embedding)}")
    else:
        print(f"错误: {embed_result.error}")

    # 流式聊天
    print("\n流式输出:")
    for chunk in stream_chat(config, "分析新能源汽车板块", client=client):
        print(chunk, end='', flush=True)

    print("\n\nFinished")
