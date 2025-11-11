# HTTP API vs Python SDK 对比分析

## 当前实现：HTTP API 方式

```python
# 使用 requests 直接调用 HTTP API
response = requests.post(
    f"{base_url}/api/chat",
    json=payload,
    timeout=60
)
```

### ✅ 优点

1. **最小依赖**
   - 只需 `requests`（通常已安装）
   - 不需要额外的 `ollama` 包

2. **完全控制**
   - 可以自定义请求头、超时、重试逻辑
   - 可以精确控制请求和响应处理

3. **透明性**
   - 直接看到 HTTP 请求/响应
   - 易于调试（可用 curl/Postman 测试）

4. **兼容性**
   - 适用于任何支持 HTTP 的客户端
   - 不依赖特定 SDK 版本

5. **灵活性**
   - 可以轻松切换不同的 LLM 服务（LM Studio, Text Generation WebUI）
   - 统一的接口设计

### ❌ 缺点

1. **代码冗长**
   - 需要手动构建 payload
   - 需要手动解析响应
   - 错误处理代码较多

2. **缺少类型提示**
   - 没有 IDE 自动补全
   - 容易出错（参数名、格式等）

3. **功能有限**
   - 流式处理需要自己实现
   - 没有内置的重试机制

---

## Python SDK 方式

```python
# 使用 ollama 包
from ollama import chat

response = chat(
    model='qwen2.5:latest',
    messages=[{'role': 'user', 'content': 'Hello'}]
)
```

### ✅ 优点

1. **代码简洁**
   - API 更直观易用
   - 更少的样板代码

2. **类型支持**
   - 有类型提示（`ChatResponse`）
   - IDE 自动补全和类型检查

3. **内置功能**
   - 流式处理：`stream=True`
   - 自动重试和错误处理
   - 连接池管理

4. **官方维护**
   - 与 Ollama 更新同步
   - 新功能自动支持

5. **更好的错误处理**
   - 结构化的异常类型
   - 更清晰的错误信息

### ❌ 缺点

1. **额外依赖**
   - 需要安装 `ollama` 包
   - 增加项目依赖

2. **抽象层**
   - 隐藏了底层 HTTP 细节
   - 调试时不够直观

3. **版本绑定**
   - 依赖特定 SDK 版本
   - 可能与 Ollama 版本不匹配

---

## 代码对比

### HTTP API 方式（当前）

```python
def call_ollama_model(self, model: str, prompt: str, 
                     system_prompt: Optional[str] = None) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 2000}
    }
    if system_prompt:
        payload["system"] = system_prompt
    
    response = requests.post(
        f"{self.base_url}/api/chat",
        json=payload,
        timeout=60
    )
    
    if response.status_code == 200:
        result = response.json()
        return result["message"]["content"]
    else:
        return f"错误: HTTP {response.status_code}"
```

### Python SDK 方式

```python
import ollama

def call_ollama_model(self, model: str, prompt: str,
                     system_prompt: Optional[str] = None) -> str:
    messages = [{"role": "user", "content": prompt}]
    
    response = ollama.chat(
        model=model,
        messages=messages,
        system=system_prompt,
        options={"temperature": 0.7, "num_predict": 2000}
    )
    
    return response["message"]["content"]
```

**代码行数对比**：HTTP API ~15 行 vs SDK ~8 行

---

## 流式处理对比

### HTTP API（需要自己实现）

```python
def stream_chat(self, model: str, prompt: str):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True
    }
    
    response = requests.post(
        f"{self.base_url}/api/chat",
        json=payload,
        stream=True,
        timeout=60
    )
    
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            if "message" in chunk:
                yield chunk["message"]["content"]
```

### Python SDK（内置支持）

```python
def stream_chat(self, model: str, prompt: str):
    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    for chunk in stream:
        yield chunk["message"]["content"]
```

---

## 推荐建议

### 使用 HTTP API 的场景

1. ✅ **最小化依赖**：不想增加额外的包
2. ✅ **多服务支持**：需要统一接口支持多个 LLM 服务
3. ✅ **完全控制**：需要自定义请求/响应处理
4. ✅ **调试需求**：需要直接查看 HTTP 请求/响应

### 使用 Python SDK 的场景

1. ✅ **开发效率**：想要更简洁的代码
2. ✅ **类型安全**：需要类型提示和 IDE 支持
3. ✅ **流式处理**：需要流式输出功能
4. ✅ **生产环境**：需要更好的错误处理和重试机制

---

## 混合方案建议

可以同时支持两种方式，让用户选择：

```python
class LocalLLMConfig:
    def __init__(self, use_sdk: bool = False):
        self.use_sdk = use_sdk
        if use_sdk:
            try:
                import ollama
                self.ollama_client = ollama.Client()
            except ImportError:
                print("警告: ollama 包未安装，回退到 HTTP API")
                self.use_sdk = False
    
    def call_ollama_model(self, ...):
        if self.use_sdk:
            return self._call_with_sdk(...)
        else:
            return self._call_with_http(...)
```

---

## 结论

**对于当前项目**，建议：

1. **保持 HTTP API 方式**（如果依赖最小化是优先考虑）
   - 项目已经支持多个 LLM 服务（Ollama, LM Studio, Text Generation WebUI）
   - 统一的 HTTP 接口更容易维护

2. **或者切换到 Python SDK**（如果开发体验优先）
   - 代码更简洁
   - 更好的类型支持
   - 内置流式处理

3. **最佳方案**：提供两种方式，让用户选择
   - 默认使用 HTTP API（兼容性好）
   - 可选使用 SDK（如果安装了 ollama 包）

