#!/usr/bin/env python3
"""
Milvus 数据库操作工具
提供常用的 Milvus 数据库操作功能，采用函数式编程风格
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from functools import partial
from datetime import datetime

from pymilvus import MilvusClient, DataType, Collection, utility, connections


# ============================================================================
# 数据类型定义
# ============================================================================

@dataclass(frozen=True)
class MilvusConfig:
    """Milvus 连接配置"""
    uri: str = "./milvus_demo.db"
    alias: str = "default"


@dataclass(frozen=True)
class CollectionInfo:
    """Collection 信息"""
    name: str
    description: str
    num_entities: int
    schema: Dict[str, Any]
    index_info: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class SearchConfig:
    """搜索配置"""
    metric_type: str = "COSINE"
    index_type: str = "HNSW"
    ef: int = 10
    nprobe: int = 10


@dataclass(frozen=True)
class OperationResult:
    """操作结果"""
    success: bool
    message: str
    data: Any = None
    error: Optional[str] = None


# ============================================================================
# 纯函数 - 数据转换
# ============================================================================

def format_collection_info(collection_name: str, stats: Dict[str, Any]) -> str:
    """格式化 collection 信息 - 纯函数"""
    info = [
        f"Collection: {collection_name}",
        f"Entity Count: {stats.get('row_count', 0)}",
    ]
    return "\n".join(info)


def format_search_results(results: List[Dict[str, Any]], limit: int = 10) -> str:
    """格式化搜索结果 - 纯函数"""
    if not results:
        return "No results found"

    output = []
    for i, result in enumerate(results[:limit], 1):
        score = result.get('distance', 0)
        text = result.get('entity', {}).get('text', '')[:100]
        output.append(f"{i}. Score: {score:.4f} | Text: {text}...")

    return "\n".join(output)


def create_schema_field(
    field_name: str,
    datatype: DataType,
    is_primary: bool = False,
    auto_id: bool = False,
    max_length: Optional[int] = None,
    dim: Optional[int] = None
) -> Dict[str, Any]:
    """创建 schema 字段定义 - 纯函数"""
    field_def = {
        "field_name": field_name,
        "datatype": datatype,
        "is_primary": is_primary,
        "auto_id": auto_id
    }

    if max_length is not None:
        field_def["max_length"] = max_length

    if dim is not None:
        field_def["dim"] = dim

    return field_def


# ============================================================================
# 连接管理
# ============================================================================

def create_client(config: MilvusConfig) -> MilvusClient:
    """创建 Milvus 客户端 - 副作用（创建连接）"""
    return MilvusClient(uri=config.uri)


def check_connection(client: MilvusClient) -> bool:
    """检查连接状态 - 副作用（网络请求）"""
    try:
        # 尝试列出 collections 来验证连接
        client.list_collections()
        return True
    except Exception:
        return False


# ============================================================================
# Collection 操作
# ============================================================================

def list_collections(client: MilvusClient) -> List[str]:
    """列出所有 collections - 副作用（查询数据库）"""
    try:
        return client.list_collections()
    except Exception as e:
        print(f"Error listing collections: {e}")
        return []


def has_collection(client: MilvusClient, collection_name: str) -> bool:
    """检查 collection 是否存在 - 副作用（查询数据库）"""
    try:
        return client.has_collection(collection_name)
    except Exception as e:
        print(f"Error checking collection: {e}")
        return False


def get_collection_stats(client: MilvusClient, collection_name: str) -> Dict[str, Any]:
    """获取 collection 统计信息 - 副作用（查询数据库）"""
    try:
        stats = client.get_collection_stats(collection_name)
        return stats
    except Exception as e:
        print(f"Error getting collection stats: {e}")
        return {}


def describe_collection(client: MilvusClient, collection_name: str) -> Dict[str, Any]:
    """描述 collection - 副作用（查询数据库）"""
    try:
        return client.describe_collection(collection_name)
    except Exception as e:
        print(f"Error describing collection: {e}")
        return {}


def create_simple_collection(
    client: MilvusClient,
    collection_name: str,
    dimension: int,
    metric_type: str = "COSINE",
    index_type: str = "HNSW"
) -> OperationResult:
    """
    创建简单的 collection
    副作用（创建数据库对象）

    Args:
        client: Milvus 客户端
        collection_name: collection 名称
        dimension: 向量维度
        metric_type: 相似度度量类型 (COSINE, L2, IP)
        index_type: 索引类型 (HNSW, IVF_FLAT, FLAT)

    Returns:
        OperationResult: 操作结果
    """
    try:
        if has_collection(client, collection_name):
            return OperationResult(
                success=False,
                message=f"Collection {collection_name} already exists",
                error="Collection exists"
            )

        # 定义 Schema
        schema = MilvusClient.create_schema(
            auto_id=True,
            enable_dynamic_field=True
        )

        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)

        # 定义 Index
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type=index_type,
            metric_type=metric_type,
            index_name="vector_index",
            params={"M": 64, "efConstruction": 100} if index_type == "HNSW" else {}
        )

        # 创建 collection
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level="Bounded"
        )

        # 加载到内存
        client.load_collection(collection_name=collection_name)

        return OperationResult(
            success=True,
            message=f"Collection {collection_name} created successfully",
            data={"collection_name": collection_name, "dimension": dimension}
        )

    except Exception as e:
        return OperationResult(
            success=False,
            message="Failed to create collection",
            error=str(e)
        )


def drop_collection(client: MilvusClient, collection_name: str) -> OperationResult:
    """
    删除 collection
    副作用（删除数据库对象）
    """
    try:
        if not has_collection(client, collection_name):
            return OperationResult(
                success=False,
                message=f"Collection {collection_name} does not exist",
                error="Collection not found"
            )

        client.drop_collection(collection_name)

        return OperationResult(
            success=True,
            message=f"Collection {collection_name} dropped successfully"
        )

    except Exception as e:
        return OperationResult(
            success=False,
            message="Failed to drop collection",
            error=str(e)
        )


def load_collection(client: MilvusClient, collection_name: str) -> OperationResult:
    """
    加载 collection 到内存
    副作用（修改数据库状态）
    """
    try:
        client.load_collection(collection_name)

        return OperationResult(
            success=True,
            message=f"Collection {collection_name} loaded successfully"
        )

    except Exception as e:
        return OperationResult(
            success=False,
            message="Failed to load collection",
            error=str(e)
        )


def release_collection(client: MilvusClient, collection_name: str) -> OperationResult:
    """
    从内存中释放 collection
    副作用（修改数据库状态）
    """
    try:
        client.release_collection(collection_name)

        return OperationResult(
            success=True,
            message=f"Collection {collection_name} released successfully"
        )

    except Exception as e:
        return OperationResult(
            success=False,
            message="Failed to release collection",
            error=str(e)
        )


# ============================================================================
# 数据操作
# ============================================================================

def insert_data(
    client: MilvusClient,
    collection_name: str,
    data: List[Dict[str, Any]]
) -> OperationResult:
    """
    插入数据
    副作用（写入数据库）

    Args:
        client: Milvus 客户端
        collection_name: collection 名称
        data: 数据列表，每个元素是一个字典

    Returns:
        OperationResult: 操作结果
    """
    try:
        if not has_collection(client, collection_name):
            return OperationResult(
                success=False,
                message=f"Collection {collection_name} does not exist",
                error="Collection not found"
            )

        if not data:
            return OperationResult(
                success=False,
                message="No data to insert",
                error="Empty data"
            )

        result = client.insert(collection_name=collection_name, data=data)

        return OperationResult(
            success=True,
            message=f"Inserted {len(data)} records into {collection_name}",
            data={"insert_count": len(data), "ids": result.get("ids", [])}
        )

    except Exception as e:
        return OperationResult(
            success=False,
            message="Failed to insert data",
            error=str(e)
        )


def query_data(
    client: MilvusClient,
    collection_name: str,
    filter_expr: str,
    output_fields: Optional[List[str]] = None,
    limit: int = 10
) -> OperationResult:
    """
    查询数据
    副作用（查询数据库）

    Args:
        client: Milvus 客户端
        collection_name: collection 名称
        filter_expr: 过滤表达式，例如 'id > 0'
        output_fields: 要返回的字段列表
        limit: 返回结果数量限制

    Returns:
        OperationResult: 操作结果
    """
    try:
        if not has_collection(client, collection_name):
            return OperationResult(
                success=False,
                message=f"Collection {collection_name} does not exist",
                error="Collection not found"
            )

        results = client.query(
            collection_name=collection_name,
            filter=filter_expr,
            output_fields=output_fields or ["*"],
            limit=limit
        )

        return OperationResult(
            success=True,
            message=f"Query returned {len(results)} results",
            data=results
        )

    except Exception as e:
        return OperationResult(
            success=False,
            message="Failed to query data",
            error=str(e)
        )


def search_vectors(
    client: MilvusClient,
    collection_name: str,
    query_vectors: List[List[float]],
    limit: int = 5,
    output_fields: Optional[List[str]] = None,
    filter_expr: Optional[str] = None,
    search_params: Optional[Dict[str, Any]] = None
) -> OperationResult:
    """
    向量搜索
    副作用（查询数据库）

    Args:
        client: Milvus 客户端
        collection_name: collection 名称
        query_vectors: 查询向量列表
        limit: 每个查询返回的结果数量
        output_fields: 要返回的字段列表
        filter_expr: 过滤表达式
        search_params: 搜索参数

    Returns:
        OperationResult: 操作结果
    """
    try:
        if not has_collection(client, collection_name):
            return OperationResult(
                success=False,
                message=f"Collection {collection_name} does not exist",
                error="Collection not found"
            )

        default_search_params = {"params": {"ef": 10}}
        params = search_params or default_search_params

        results = client.search(
            collection_name=collection_name,
            data=query_vectors,
            anns_field="vector",
            limit=limit,
            output_fields=output_fields or ["text"],
            filter=filter_expr,
            search_params=params
        )

        return OperationResult(
            success=True,
            message=f"Search completed, found {len(results)} result sets",
            data=results
        )

    except Exception as e:
        return OperationResult(
            success=False,
            message="Failed to search vectors",
            error=str(e)
        )


def delete_data(
    client: MilvusClient,
    collection_name: str,
    ids: Optional[List[int]] = None,
    filter_expr: Optional[str] = None
) -> OperationResult:
    """
    删除数据
    副作用（修改数据库）

    Args:
        client: Milvus 客户端
        collection_name: collection 名称
        ids: 要删除的 ID 列表
        filter_expr: 过滤表达式

    Returns:
        OperationResult: 操作结果
    """
    try:
        if not has_collection(client, collection_name):
            return OperationResult(
                success=False,
                message=f"Collection {collection_name} does not exist",
                error="Collection not found"
            )

        if ids:
            client.delete(collection_name=collection_name, ids=ids)
            message = f"Deleted {len(ids)} records by IDs"
        elif filter_expr:
            client.delete(collection_name=collection_name, filter=filter_expr)
            message = f"Deleted records matching filter: {filter_expr}"
        else:
            return OperationResult(
                success=False,
                message="Must provide either ids or filter_expr",
                error="Invalid parameters"
            )

        return OperationResult(
            success=True,
            message=message
        )

    except Exception as e:
        return OperationResult(
            success=False,
            message="Failed to delete data",
            error=str(e)
        )


# ============================================================================
# 便捷函数
# ============================================================================

def get_collection_info(client: MilvusClient, collection_name: str) -> CollectionInfo:
    """
    获取完整的 collection 信息
    副作用（查询数据库）
    """
    try:
        if not has_collection(client, collection_name):
            raise ValueError(f"Collection {collection_name} does not exist")

        desc = describe_collection(client, collection_name)
        stats = get_collection_stats(client, collection_name)

        return CollectionInfo(
            name=collection_name,
            description=desc.get("description", ""),
            num_entities=stats.get("row_count", 0),
            schema=desc.get("schema", {}),
            index_info=desc.get("indexes", [])
        )

    except Exception as e:
        print(f"Error getting collection info: {e}")
        return CollectionInfo(
            name=collection_name,
            description="Error",
            num_entities=0,
            schema={}
        )


def print_all_collections(client: MilvusClient) -> None:
    """
    打印所有 collections 的信息
    副作用（查询数据库 + 输出）
    """
    collections = list_collections(client)

    if not collections:
        print("No collections found")
        return

    print(f"\n{'='*60}")
    print(f"Found {len(collections)} collection(s):")
    print(f"{'='*60}\n")

    for coll_name in collections:
        try:
            stats = get_collection_stats(client, coll_name)
            row_count = stats.get("row_count", 0)
            print(f"📦 {coll_name}")
            print(f"   └─ Entities: {row_count:,}")
            print()
        except Exception as e:
            print(f"📦 {coll_name}")
            print(f"   └─ Error: {e}")
            print()


def backup_collection_data(
    client: MilvusClient,
    collection_name: str,
    output_file: str,
    batch_size: int = 1000
) -> OperationResult:
    """
    备份 collection 数据到 JSON 文件
    副作用（查询数据库 + 写入文件）

    Args:
        client: Milvus 客户端
        collection_name: collection 名称
        output_file: 输出文件路径
        batch_size: 批次大小

    Returns:
        OperationResult: 操作结果
    """
    try:
        if not has_collection(client, collection_name):
            return OperationResult(
                success=False,
                message=f"Collection {collection_name} does not exist",
                error="Collection not found"
            )

        # 查询所有数据
        all_data = []
        offset = 0

        while True:
            result = query_data(
                client,
                collection_name,
                filter_expr="id >= 0",
                limit=batch_size
            )

            if not result.success or not result.data:
                break

            all_data.extend(result.data)
            offset += batch_size

            if len(result.data) < batch_size:
                break

        # 保存到文件
        backup_data = {
            "collection_name": collection_name,
            "timestamp": datetime.now().isoformat(),
            "count": len(all_data),
            "data": all_data
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)

        return OperationResult(
            success=True,
            message=f"Backed up {len(all_data)} records to {output_file}",
            data={"count": len(all_data), "file": output_file}
        )

    except Exception as e:
        return OperationResult(
            success=False,
            message="Failed to backup collection",
            error=str(e)
        )


# ============================================================================
# 主函数示例
# ============================================================================

def main():
    """演示 Milvus 工具的使用"""

    print("="*60)
    print("Milvus Database Tool - Demo")
    print("="*60)

    # 创建配置和客户端
    config = MilvusConfig(uri="./milvus_demo.db")
    client = create_client(config)

    # 检查连接
    if check_connection(client):
        print("✅ Connected to Milvus\n")
    else:
        print("❌ Failed to connect to Milvus\n")
        return

    # 列出所有 collections
    print_all_collections(client)

    # 示例：创建一个新的 collection
    test_collection = "test_collection_demo"

    print(f"\n{'='*60}")
    print(f"Creating test collection: {test_collection}")
    print(f"{'='*60}\n")

    result = create_simple_collection(
        client,
        collection_name=test_collection,
        dimension=128,
        metric_type="COSINE",
        index_type="HNSW"
    )

    print(f"Result: {result.message}")

    if result.success:
        # 插入测试数据
        import random

        test_data = [
            {
                "vector": [random.random() for _ in range(128)],
                "text": f"This is test document {i}",
                "metadata": {"doc_id": i}
            }
            for i in range(10)
        ]

        print(f"\nInserting {len(test_data)} test records...")
        insert_result = insert_data(client, test_collection, test_data)
        print(f"Result: {insert_result.message}")

        # 查询数据
        print("\nQuerying data...")
        query_result = query_data(
            client,
            test_collection,
            filter_expr="id >= 0",
            limit=5
        )
        print(f"Result: {query_result.message}")
        if query_result.data:
            print(f"Sample record: {query_result.data[0]}")

        # 向量搜索
        print("\nPerforming vector search...")
        query_vector = [random.random() for _ in range(128)]
        search_result = search_vectors(
            client,
            test_collection,
            query_vectors=[query_vector],
            limit=3
        )
        print(f"Result: {search_result.message}")

        # 删除 collection
        print(f"\nCleaning up: dropping {test_collection}...")
        drop_result = drop_collection(client, test_collection)
        print(f"Result: {drop_result.message}")

    print(f"\n{'='*60}")
    print("Demo completed")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
