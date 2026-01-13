# 函数式编程版本 - 投资研究支持系统

这是原有系统的函数式编程重构版本，位于 `src_functional` 目录中。

## 📖 目录

- [概述](#概述)
- [函数式编程原则](#函数式编程原则)
- [主要改进点](#主要改进点)
- [文件结构](#文件结构)
- [使用方法](#使用方法)
- [对比示例](#对比示例)
- [优势与权衡](#优势与权衡)

## 概述

本版本将原有的面向对象代码重构为函数式编程风格，遵循函数式编程的核心原则，提高代码的可测试性、可组合性和可维护性。

## 函数式编程原则

### 1. 不可变数据结构 (Immutability)

使用 `@dataclass(frozen=True)` 创建不可变的数据类型：

```python
@dataclass(frozen=True)
class LLMConfig:
    """不可变的 LLM 配置"""
    model: str
    embed_model: str
    temperature: float = 0.8
    num_predict: int = 2000
```

### 2. 纯函数 (Pure Functions)

明确区分纯函数和有副作用的函数：

```python
# 纯函数 - 无副作用
def build_messages(
    prompt: str,
    system_prompt: Optional[str] = None,
    history: Optional[List[Message]] = None
) -> List[Message]:
    """构建消息列表 - 纯函数"""
    messages = []
    if system_prompt:
        messages.append(create_system_message(system_prompt))
    if history:
        messages.extend(history)
    messages.append(create_user_message(prompt))
    return messages
```

### 3. 函数组合 (Function Composition)

使用高阶函数和函数组合：

```python
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
```

### 4. 高阶函数 (Higher-Order Functions)

使用装饰器和高阶函数处理横切关注点：

```python
def with_error_handling(func: Callable) -> Callable:
    """高阶函数：为函数添加错误处理装饰器"""
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            if 'embed' in func.__name__:
                return EmbeddingResult(embedding=[], error=str(e))
            else:
                return ChatResult(content="", error=str(e))
    return wrapper
```

### 5. 偏函数应用 (Partial Application)

使用 `functools.partial` 创建预配置的函数：

```python
def create_chat_function(config: LLMConfig, client: Optional[ollama.Client] = None) -> Callable:
    """创建一个预配置的聊天函数"""
    return partial(simple_chat, config=config, client=client)

# 使用
chat_fn = create_chat_function(config)
result = chat_fn(prompt="你好")
```

## 主要改进点

### 1. 数据与行为分离

**原版（面向对象）：**
```python
class DirectOllamaLLM:
    def __init__(self, model: str, embed_model: str):
        self.model = model
        self.embed_model = embed_model
        self.client = ollama.Client()

    def simple_chat(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        # 使用 self.model, self.client
        pass
```

**函数式版本：**
```python
@dataclass(frozen=True)
class LLMConfig:
    model: str
    embed_model: str
    temperature: float = 0.8
    num_predict: int = 2000

def simple_chat(
    config: LLMConfig,
    prompt: str,
    system_prompt: Optional[str] = None,
    client: Optional[ollama.Client] = None
) -> ChatResult:
    # 配置作为参数传递
    pass
```

### 2. 明确的副作用管理

函数式版本明确标识有副作用的函数（使用 `_` 前缀或注释）：

```python
# 副作用函数 - 与外部系统交互
def _call_ollama_client(client: ollama.Client) -> Callable:
    """返回一个与 Ollama 客户端交互的函数（柯里化）"""
    def call_chat(config: LLMConfig, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """调用 Ollama 聊天 API（副作用）"""
        return ollama.chat(model=config.model, messages=messages)
    return call_chat
```

### 3. 更好的类型安全

使用明确的返回类型和不可变数据结构：

```python
@dataclass(frozen=True)
class ChatResult:
    """不可变的聊天结果"""
    content: str
    error: Optional[str] = None

    @property
    def is_success(self) -> bool:
        return self.error is None
```

### 4. 函数可组合性

函数式版本更容易组合和重用：

```python
# 创建专门的搜索函数
search_fn = partial(
    search_kb,
    client=client,
    collection_name=collection_name,
    llm_config=llm_config,
    limit=2
)

# 可以传递给其他函数或映射到列表
results = [search_fn(query=q) for q in queries]
```

### 5. 更容易测试

纯函数更容易测试，不需要 mock 对象：

```python
# 测试纯函数
def test_build_messages():
    messages = build_messages("Hello", "You are helpful")
    assert len(messages) == 2
    assert messages[0].role == "system"
    assert messages[1].role == "user"
```

## 文件结构

```
src_functional/
├── README.md                    # 本文档
├── local_llm_direct.py         # 函数式 LLM 接口
├── local_llm_config.py         # 函数式配置管理
├── knowledge_base.py           # 函数式知识库
└── main.py                     # 函数式演示脚本
```

## 使用方法

### 1. 本地 LLM 使用

```python
from local_llm_direct import (
    create_config,
    simple_chat,
    stream_chat,
    embed_text,
    create_chat_function
)

# 创建配置
config = create_config(
    model='qwen3:0.6b',
    embed_model='qwen3-embedding:0.6b'
)

# 方式 1: 直接调用
result = simple_chat(
    config=config,
    prompt="简述中国股市特点",
    system_prompt="你是专业的金融分析师"
)

if result.is_success:
    print(result.content)
else:
    print(f"错误: {result.error}")

# 方式 2: 创建预配置函数
chat_fn = create_chat_function(config)
result = chat_fn(prompt="你好")

# 流式输出
for chunk in stream_chat(config, "分析新能源汽车板块"):
    print(chunk, end='', flush=True)
```

### 2. 知识库使用

```python
from knowledge_base import (
    create_kb_context,
    add_document_to_kb,
    search_kb,
    scan_and_add_directory,
    load_file_hashes,
    save_file_hashes
)

# 创建知识库上下文
client, coll_config, llm_config, proc_config = create_kb_context(
    collection_name="my_knowledge_base"
)

# 加载文件哈希
file_hashes = load_file_hashes()

# 添加单个文档
success, new_hashes = add_document_to_kb(
    client,
    coll_config.collection_name,
    "path/to/document.pdf",
    llm_config,
    proc_config,
    file_hashes
)

# 或扫描整个目录
new_hashes = scan_and_add_directory(
    client,
    coll_config.collection_name,
    "documents",
    llm_config,
    proc_config,
    file_hashes
)

# 保存哈希
save_file_hashes(new_hashes)

# 搜索
results = search_kb(
    client,
    coll_config.collection_name,
    "今年国庆节消费怎么样？",
    llm_config,
    limit=5
)

for result in results:
    print(f"{result.text[:100]}... (相关度: {result.score:.3f})")
```

### 3. 运行演示

```bash
cd src_functional
python main.py
```

## 对比示例

### 示例 1: 聊天功能

**原版（面向对象）：**
```python
# 需要创建实例
llm = DirectOllamaLLM(model='qwen3:0.6b', embed_model='qwen3-embedding:0.6b')
response = llm.simple_chat("你好", "你是助手")

# 难以组合和传递
def process_queries(queries, llm):
    return [llm.simple_chat(q) for q in queries]
```

**函数式版本：**
```python
# 配置和执行分离
config = create_config(model='qwen3:0.6b', embed_model='qwen3-embedding:0.6b')
result = simple_chat(config, "你好", "你是助手")

# 易于组合
chat_fn = create_chat_function(config)
results = [chat_fn(prompt=q) for q in queries]

# 易于传递给高阶函数
results = list(map(lambda q: chat_fn(prompt=q), queries))
```

### 示例 2: 知识库操作

**原版（面向对象）：**
```python
# 需要创建实例并维护状态
kb = KnowledgeBase(collection_name="my_kb")
kb.add_document("doc.pdf")
results = kb.search("query")

# 状态耦合，难以并行处理
for file in files:
    kb.add_document(file)  # 每次调用都修改 kb 的状态
```

**函数式版本：**
```python
# 无状态，所有依赖作为参数传递
client, coll_config, llm_config, proc_config = create_kb_context()
file_hashes = load_file_hashes()

# 返回新的哈希字典，不修改原始数据
success, new_hashes = add_document_to_kb(
    client, coll_config.collection_name, "doc.pdf",
    llm_config, proc_config, file_hashes
)

# 易于并行处理（如果需要）
from concurrent.futures import ThreadPoolExecutor

def add_doc(file_path):
    return add_document_to_kb(
        client, coll_config.collection_name, file_path,
        llm_config, proc_config, file_hashes
    )

with ThreadPoolExecutor() as executor:
    results = list(executor.map(add_doc, files))
```

## 优势与权衡

### 优势

1. **更好的测试性**
   - 纯函数易于单元测试
   - 不需要复杂的 mock 对象
   - 测试更加隔离和可靠

2. **更容易理解**
   - 函数签名明确说明输入和输出
   - 无隐藏状态，数据流清晰
   - 副作用被明确标识

3. **更好的可组合性**
   - 函数可以自由组合
   - 使用高阶函数和偏函数应用
   - 易于创建 DSL（领域特定语言）

4. **更安全的并发**
   - 不可变数据结构天然线程安全
   - 纯函数无竞态条件
   - 易于并行处理

5. **更容易推理**
   - 函数行为由输入完全决定
   - 无需追踪对象状态变化
   - 减少认知负担

### 权衡

1. **学习曲线**
   - 需要理解函数式编程概念
   - 习惯面向对象的开发者需要适应

2. **性能考虑**
   - 不可变数据可能增加内存使用
   - 某些场景下需要复制数据

3. **Python 限制**
   - Python 不是纯函数式语言
   - 某些函数式特性需要额外的库或模式

4. **代码量**
   - 某些情况下函数式代码可能更冗长
   - 需要更多的类型定义

## 最佳实践

### 1. 使用类型提示

```python
from typing import List, Dict, Any, Optional, Callable

def process_data(
    data: List[str],
    transform: Callable[[str], str],
    filter_fn: Optional[Callable[[str], bool]] = None
) -> List[str]:
    """明确的类型提示提高代码可读性"""
    pass
```

### 2. 明确标识副作用

```python
# 纯函数
def calculate_score(data: Dict[str, Any]) -> float:
    """计算分数 - 纯函数"""
    pass

# 副作用函数
def save_to_database(data: Dict[str, Any]) -> bool:
    """保存到数据库 - 副作用（IO 操作）"""
    pass
```

### 3. 使用偏函数应用创建专用函数

```python
# 创建专用的搜索函数
search_finance = partial(search_kb, collection_name="finance_kb")
search_tech = partial(search_kb, collection_name="tech_kb")

# 使用
finance_results = search_finance(query="股市")
tech_results = search_tech(query="AI")
```

### 4. 组合小函数构建大函数

```python
# 小的纯函数
def validate_input(data: str) -> bool:
    return len(data) > 0

def normalize_text(text: str) -> str:
    return text.strip().lower()

def tokenize(text: str) -> List[str]:
    return text.split()

# 组合成大函数
def process_text(text: str) -> Optional[List[str]]:
    if not validate_input(text):
        return None
    normalized = normalize_text(text)
    return tokenize(normalized)

# 或使用 pipe
process = pipe(normalize_text, tokenize)
tokens = process(text)
```

## 总结

函数式编程版本提供了更清晰、更可测试、更可组合的代码结构。虽然有一定的学习曲线，但长期来看，这种风格能够提高代码质量和可维护性。

建议：
- 新项目优先考虑函数式风格
- 现有项目可以逐步重构关键模块
- 根据团队熟悉度和项目需求选择合适的编程范式
- 两种风格可以混合使用，在合适的场景使用合适的方法
