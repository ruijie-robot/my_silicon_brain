# Milvus 数据库操作工具

一个功能完整的 Milvus 向量数据库操作工具，采用函数式编程风格设计。

## 📖 目录

- [简介](#简介)
- [特性](#特性)
- [安装](#安装)
- [快速开始](#快速开始)
- [API 文档](#api-文档)
- [使用示例](#使用示例)
- [最佳实践](#最佳实践)

## 简介

`milvus_tool.py` 提供了一套简洁、易用的 Milvus 数据库操作接口，包括：

- Collection 管理（创建、删除、查询）
- 数据操作（插入、查询、删除）
- 向量搜索
- 数据备份和恢复
- 内存管理

所有函数都采用函数式编程风格，明确区分纯函数和副作用函数。

## 特性

✅ **函数式设计** - 纯函数和副作用分离，易于测试和组合
✅ **类型安全** - 使用 dataclass 和类型提示
✅ **错误处理** - 统一的 OperationResult 返回类型
✅ **易于使用** - 简洁的 API 设计
✅ **功能完整** - 涵盖所有常用操作
✅ **文档齐全** - 详细的注释和示例

## 安装

确保已安装必要的依赖：

```bash
pip install pymilvus
```

## 快速开始

### 1. 基础使用

```python
from milvus_tool import (
    MilvusConfig,
    create_client,
    list_collections,
    print_all_collections
)

# 创建配置和客户端
config = MilvusConfig(uri="./milvus_demo.db")
client = create_client(config)

# 列出所有 collections
collections = list_collections(client)
print(f"Collections: {collections}")

# 打印详细信息
print_all_collections(client)
```

### 2. 创建 Collection

```python
from milvus_tool import create_simple_collection

result = create_simple_collection(
    client,
    collection_name="my_vectors",
    dimension=128,
    metric_type="COSINE",
    index_type="HNSW"
)

if result.success:
    print(f"✅ {result.message}")
else:
    print(f"❌ {result.error}")
```

### 3. 插入数据

```python
from milvus_tool import insert_data

data = [
    {
        "vector": [0.1, 0.2, 0.3, ...],  # 128 维向量
        "text": "这是一段文本",
        "metadata": {"source": "doc1"}
    },
    # 更多数据...
]

result = insert_data(client, "my_vectors", data)
print(f"插入了 {result.data['insert_count']} 条记录")
```

### 4. 向量搜索

```python
from milvus_tool import search_vectors

query_vector = [0.1, 0.2, 0.3, ...]  # 查询向量

result = search_vectors(
    client,
    collection_name="my_vectors",
    query_vectors=[query_vector],
    limit=5,
    output_fields=["text", "metadata"]
)

if result.success:
    for result_set in result.data:
        for item in result_set:
            print(f"相似度: {item['distance']:.4f}")
            print(f"文本: {item['entity']['text']}")
```

### 5. 查询数据

```python
from milvus_tool import query_data

result = query_data(
    client,
    collection_name="my_vectors",
    filter_expr='metadata["source"] == "doc1"',
    output_fields=["text", "metadata"],
    limit=10
)

for record in result.data:
    print(record)
```

## API 文档

### 数据类型

#### MilvusConfig

```python
@dataclass(frozen=True)
class MilvusConfig:
    uri: str = "./milvus_demo.db"  # Milvus 连接 URI
    alias: str = "default"          # 连接别名
```

#### OperationResult

```python
@dataclass(frozen=True)
class OperationResult:
    success: bool          # 操作是否成功
    message: str          # 操作消息
    data: Any = None      # 返回数据
    error: Optional[str] = None  # 错误信息
```

### 连接管理

#### create_client

```python
def create_client(config: MilvusConfig) -> MilvusClient
```

创建 Milvus 客户端。

**参数：**
- `config`: Milvus 配置对象

**返回：**
- `MilvusClient`: Milvus 客户端实例

#### check_connection

```python
def check_connection(client: MilvusClient) -> bool
```

检查连接状态。

**返回：**
- `bool`: 连接是否正常

### Collection 操作

#### create_simple_collection

```python
def create_simple_collection(
    client: MilvusClient,
    collection_name: str,
    dimension: int,
    metric_type: str = "COSINE",
    index_type: str = "HNSW"
) -> OperationResult
```

创建简单的 collection。

**参数：**
- `client`: Milvus 客户端
- `collection_name`: collection 名称
- `dimension`: 向量维度
- `metric_type`: 相似度度量类型 (COSINE, L2, IP)
- `index_type`: 索引类型 (HNSW, IVF_FLAT, FLAT)

**返回：**
- `OperationResult`: 操作结果

#### list_collections

```python
def list_collections(client: MilvusClient) -> List[str]
```

列出所有 collections。

**返回：**
- `List[str]`: collection 名称列表

#### drop_collection

```python
def drop_collection(client: MilvusClient, collection_name: str) -> OperationResult
```

删除 collection。

**返回：**
- `OperationResult`: 操作结果

#### get_collection_info

```python
def get_collection_info(client: MilvusClient, collection_name: str) -> CollectionInfo
```

获取完整的 collection 信息。

**返回：**
- `CollectionInfo`: collection 详细信息

### 数据操作

#### insert_data

```python
def insert_data(
    client: MilvusClient,
    collection_name: str,
    data: List[Dict[str, Any]]
) -> OperationResult
```

插入数据。

**参数：**
- `data`: 数据列表，每个元素是一个字典，必须包含 `vector` 字段

**返回：**
- `OperationResult`: 包含插入的 ID 列表

#### query_data

```python
def query_data(
    client: MilvusClient,
    collection_name: str,
    filter_expr: str,
    output_fields: Optional[List[str]] = None,
    limit: int = 10
) -> OperationResult
```

查询数据。

**参数：**
- `filter_expr`: 过滤表达式，例如 `'id > 0'` 或 `'category == "AI"'`
- `output_fields`: 要返回的字段列表
- `limit`: 返回结果数量限制

**返回：**
- `OperationResult`: 包含查询结果

#### search_vectors

```python
def search_vectors(
    client: MilvusClient,
    collection_name: str,
    query_vectors: List[List[float]],
    limit: int = 5,
    output_fields: Optional[List[str]] = None,
    filter_expr: Optional[str] = None,
    search_params: Optional[Dict[str, Any]] = None
) -> OperationResult
```

向量搜索。

**参数：**
- `query_vectors`: 查询向量列表
- `limit`: 每个查询返回的结果数量
- `output_fields`: 要返回的字段列表
- `filter_expr`: 过滤表达式（可选）
- `search_params`: 搜索参数（可选）

**返回：**
- `OperationResult`: 包含搜索结果

#### delete_data

```python
def delete_data(
    client: MilvusClient,
    collection_name: str,
    ids: Optional[List[int]] = None,
    filter_expr: Optional[str] = None
) -> OperationResult
```

删除数据。

**参数：**
- `ids`: 要删除的 ID 列表（二选一）
- `filter_expr`: 过滤表达式（二选一）

**返回：**
- `OperationResult`: 操作结果

### 实用工具

#### print_all_collections

```python
def print_all_collections(client: MilvusClient) -> None
```

打印所有 collections 的信息。

#### backup_collection_data

```python
def backup_collection_data(
    client: MilvusClient,
    collection_name: str,
    output_file: str,
    batch_size: int = 1000
) -> OperationResult
```

备份 collection 数据到 JSON 文件。

**参数：**
- `output_file`: 输出文件路径
- `batch_size`: 批次大小

**返回：**
- `OperationResult`: 包含备份信息

## 使用示例

### 完整工作流程

```python
from milvus_tool import *

# 1. 创建客户端
config = MilvusConfig(uri="./milvus_demo.db")
client = create_client(config)

# 2. 创建 collection
create_simple_collection(
    client,
    collection_name="documents",
    dimension=256,
    metric_type="COSINE"
)

# 3. 插入数据
import random

data = [
    {
        "vector": [random.random() for _ in range(256)],
        "text": f"Document {i}",
        "category": "tech" if i % 2 == 0 else "finance"
    }
    for i in range(100)
]

insert_data(client, "documents", data)

# 4. 向量搜索
query_vector = [random.random() for _ in range(256)]
search_result = search_vectors(
    client,
    "documents",
    query_vectors=[query_vector],
    limit=5,
    output_fields=["text", "category"]
)

# 5. 条件查询
query_result = query_data(
    client,
    "documents",
    filter_expr='category == "tech"',
    limit=10
)

# 6. 备份数据
backup_collection_data(
    client,
    "documents",
    output_file="backup.json"
)

# 7. 清理
drop_collection(client, "documents")
```

### 更多示例

运行示例脚本查看更多用法：

```bash
python tools/milvus_tool_example.py
```

示例包括：
- 基础操作
- Collection 管理
- 数据插入和查询
- 向量搜索
- 删除操作
- 备份和恢复
- 内存管理

## 最佳实践

### 1. 错误处理

始终检查 `OperationResult` 的 `success` 字段：

```python
result = insert_data(client, "my_collection", data)

if result.success:
    print(f"✅ {result.message}")
    # 使用 result.data
else:
    print(f"❌ 错误: {result.error}")
    # 处理错误
```

### 2. 资源管理

使用完 collection 后记得释放内存：

```python
# 释放内存
release_collection(client, "large_collection")

# 需要时再加载
load_collection(client, "large_collection")
```

### 3. 批量操作

大量数据时使用批量插入：

```python
batch_size = 1000
for i in range(0, len(all_data), batch_size):
    batch = all_data[i:i+batch_size]
    insert_data(client, "my_collection", batch)
```

### 4. 搜索优化

使用合适的搜索参数：

```python
# 精确搜索（较慢）
search_params = {"params": {"ef": 100}}

# 快速搜索（略低精度）
search_params = {"params": {"ef": 10}}

search_vectors(
    client,
    "my_collection",
    query_vectors=[query],
    search_params=search_params
)
```

### 5. 过滤表达式

使用正确的过滤语法：

```python
# 数值比较
filter_expr = "score >= 80"

# 字符串匹配
filter_expr = 'category == "AI"'

# 逻辑组合
filter_expr = "score >= 80 and category == 'AI'"

# IN 操作
filter_expr = "category in ['AI', 'ML', 'DL']"
```

## 运行测试

```bash
# 运行基础测试
python tools/milvus_tool.py

# 运行完整示例
python tools/milvus_tool_example.py
```

## 故障排查

### 连接失败

```python
if not check_connection(client):
    print("检查 Milvus 是否正在运行")
    print("URI 是否正确")
```

### Collection 不存在

```python
if not has_collection(client, "my_collection"):
    print("Collection 不存在，需要先创建")
    create_simple_collection(client, "my_collection", dimension=128)
```

### 维度不匹配

确保插入的向量维度与 collection 定义的维度一致：

```python
# Collection 定义: dimension=128
# 插入数据时向量也必须是 128 维
data = [{"vector": [random.random() for _ in range(128)], ...}]
```

## 总结

这个工具提供了一套完整、易用的 Milvus 操作接口，适合：

- 快速原型开发
- 数据探索和分析
- 自动化脚本
- 学习 Milvus 使用

采用函数式编程风格，代码清晰、易于测试和维护。

## 相关链接

- [Milvus 官方文档](https://milvus.io/docs)
- [PyMilvus API 文档](https://pymilvus.readthedocs.io/)
