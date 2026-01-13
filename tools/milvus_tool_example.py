#!/usr/bin/env python3
"""
Milvus 工具使用示例
展示常见的数据库操作场景
"""

from milvus_tool import (
    MilvusConfig,
    create_client,
    check_connection,
    list_collections,
    create_simple_collection,
    drop_collection,
    insert_data,
    query_data,
    search_vectors,
    delete_data,
    get_collection_info,
    print_all_collections,
    backup_collection_data,
    load_collection,
    release_collection
)
import random


def example_basic_operations():
    """示例 1: 基础操作"""
    print("\n" + "="*60)
    print("示例 1: 基础操作")
    print("="*60)

    # 1. 创建客户端
    config = MilvusConfig(uri="./milvus_demo.db")
    client = create_client(config)

    # 2. 检查连接
    if not check_connection(client):
        print("❌ 无法连接到 Milvus")
        return

    print("✅ 已连接到 Milvus")

    # 3. 列出所有 collections
    collections = list_collections(client)
    print(f"\n当前 collections: {collections}")

    # 4. 打印详细信息
    print_all_collections(client)


def example_create_and_manage_collection():
    """示例 2: 创建和管理 collection"""
    print("\n" + "="*60)
    print("示例 2: 创建和管理 collection")
    print("="*60)

    config = MilvusConfig(uri="./milvus_demo.db")
    client = create_client(config)

    collection_name = "my_vectors"

    # 1. 创建 collection
    print(f"\n创建 collection: {collection_name}")
    result = create_simple_collection(
        client,
        collection_name=collection_name,
        dimension=256,
        metric_type="COSINE",
        index_type="HNSW"
    )
    print(f"结果: {result.message}")

    # 2. 获取 collection 信息
    if result.success:
        info = get_collection_info(client, collection_name)
        print(f"\nCollection 信息:")
        print(f"  名称: {info.name}")
        print(f"  实体数: {info.num_entities}")
        print(f"  描述: {info.description}")

    # 3. 删除 collection
    print(f"\n删除 collection: {collection_name}")
    drop_result = drop_collection(client, collection_name)
    print(f"结果: {drop_result.message}")


def example_insert_and_query():
    """示例 3: 插入和查询数据"""
    print("\n" + "="*60)
    print("示例 3: 插入和查询数据")
    print("="*60)

    config = MilvusConfig(uri="./milvus_demo.db")
    client = create_client(config)

    collection_name = "documents"

    # 1. 创建 collection
    result = create_simple_collection(
        client,
        collection_name=collection_name,
        dimension=128
    )

    if not result.success:
        print(f"❌ {result.message}")
        return

    # 2. 准备数据
    documents = [
        {
            "vector": [random.random() for _ in range(128)],
            "text": "人工智能是计算机科学的一个分支",
            "category": "AI",
            "importance": 5
        },
        {
            "vector": [random.random() for _ in range(128)],
            "text": "机器学习是实现人工智能的一种方法",
            "category": "ML",
            "importance": 4
        },
        {
            "vector": [random.random() for _ in range(128)],
            "text": "深度学习是机器学习的一个子领域",
            "category": "DL",
            "importance": 5
        },
        {
            "vector": [random.random() for _ in range(128)],
            "text": "自然语言处理研究计算机与人类语言的交互",
            "category": "NLP",
            "importance": 4
        },
        {
            "vector": [random.random() for _ in range(128)],
            "text": "计算机视觉让计算机能够理解和处理图像",
            "category": "CV",
            "importance": 4
        }
    ]

    # 3. 插入数据
    print(f"\n插入 {len(documents)} 条记录...")
    insert_result = insert_data(client, collection_name, documents)
    print(f"结果: {insert_result.message}")

    # 4. 查询所有数据
    print("\n查询所有数据...")
    query_result = query_data(
        client,
        collection_name,
        filter_expr="id >= 0",
        output_fields=["text", "category", "importance"],
        limit=10
    )
    print(f"结果: {query_result.message}")

    if query_result.success and query_result.data:
        print("\n前 3 条记录:")
        for i, record in enumerate(query_result.data[:3], 1):
            print(f"  {i}. {record.get('text', '')}")
            print(f"     类别: {record.get('category', '')}, 重要性: {record.get('importance', 0)}")

    # 5. 条件查询
    print("\n查询重要性 >= 5 的记录...")
    filtered_result = query_data(
        client,
        collection_name,
        filter_expr="importance >= 5",
        output_fields=["text", "importance"]
    )
    print(f"结果: {filtered_result.message}")

    if filtered_result.success and filtered_result.data:
        for record in filtered_result.data:
            print(f"  - {record.get('text', '')} (重要性: {record.get('importance', 0)})")

    # 6. 清理
    print(f"\n清理: 删除 collection...")
    drop_collection(client, collection_name)


def example_vector_search():
    """示例 4: 向量搜索"""
    print("\n" + "="*60)
    print("示例 4: 向量搜索")
    print("="*60)

    config = MilvusConfig(uri="./milvus_demo.db")
    client = create_client(config)

    collection_name = "embeddings"

    # 1. 创建 collection
    result = create_simple_collection(
        client,
        collection_name=collection_name,
        dimension=64
    )

    if not result.success:
        print(f"❌ {result.message}")
        return

    # 2. 插入向量数据
    data = []
    for i in range(100):
        # 生成随机向量
        vector = [random.random() for _ in range(64)]
        data.append({
            "vector": vector,
            "text": f"Document {i}",
            "score": random.randint(1, 100)
        })

    print(f"\n插入 {len(data)} 条向量...")
    insert_data(client, collection_name, data)

    # 3. 生成查询向量
    query_vector = [random.random() for _ in range(64)]

    # 4. 执行向量搜索
    print("\n执行向量搜索...")
    search_result = search_vectors(
        client,
        collection_name,
        query_vectors=[query_vector],
        limit=5,
        output_fields=["text", "score"]
    )

    print(f"结果: {search_result.message}")

    if search_result.success and search_result.data:
        print("\n最相似的 5 条记录:")
        for i, result_set in enumerate(search_result.data):
            for j, result in enumerate(result_set, 1):
                entity = result.get('entity', {})
                distance = result.get('distance', 0)
                print(f"  {j}. {entity.get('text', '')} (相似度: {distance:.4f}, 分数: {entity.get('score', 0)})")

    # 5. 带过滤条件的搜索
    print("\n执行带过滤条件的搜索 (score >= 50)...")
    filtered_search = search_vectors(
        client,
        collection_name,
        query_vectors=[query_vector],
        limit=3,
        output_fields=["text", "score"],
        filter_expr="score >= 50"
    )

    if filtered_search.success and filtered_search.data:
        print("\n过滤后的结果:")
        for result_set in filtered_search.data:
            for i, result in enumerate(result_set, 1):
                entity = result.get('entity', {})
                distance = result.get('distance', 0)
                print(f"  {i}. {entity.get('text', '')} (相似度: {distance:.4f}, 分数: {entity.get('score', 0)})")

    # 6. 清理
    print(f"\n清理: 删除 collection...")
    drop_collection(client, collection_name)


def example_delete_operations():
    """示例 5: 删除操作"""
    print("\n" + "="*60)
    print("示例 5: 删除操作")
    print("="*60)

    config = MilvusConfig(uri="./milvus_demo.db")
    client = create_client(config)

    collection_name = "temp_data"

    # 1. 创建并插入数据
    create_simple_collection(client, collection_name, dimension=32)

    data = [
        {
            "vector": [random.random() for _ in range(32)],
            "text": f"Item {i}",
            "status": "active" if i % 2 == 0 else "inactive"
        }
        for i in range(20)
    ]

    insert_result = insert_data(client, collection_name, data)
    print(f"\n插入了 {len(data)} 条记录")

    # 获取插入的 IDs
    inserted_ids = insert_result.data.get("ids", []) if insert_result.data else []

    # 2. 查询初始状态
    initial_query = query_data(client, collection_name, filter_expr="id >= 0", limit=100)
    print(f"初始记录数: {len(initial_query.data) if initial_query.data else 0}")

    # 3. 按 ID 删除
    if inserted_ids:
        ids_to_delete = inserted_ids[:5]
        print(f"\n按 ID 删除前 5 条记录: {ids_to_delete}")
        delete_result = delete_data(client, collection_name, ids=ids_to_delete)
        print(f"结果: {delete_result.message}")

    # 4. 按条件删除
    print("\n按条件删除 status == 'inactive' 的记录...")
    delete_result = delete_data(
        client,
        collection_name,
        filter_expr='status == "inactive"'
    )
    print(f"结果: {delete_result.message}")

    # 5. 查询最终状态
    final_query = query_data(client, collection_name, filter_expr="id >= 0", limit=100)
    print(f"最终记录数: {len(final_query.data) if final_query.data else 0}")

    # 6. 清理
    print(f"\n清理: 删除 collection...")
    drop_collection(client, collection_name)


def example_backup_and_restore():
    """示例 6: 备份和恢复"""
    print("\n" + "="*60)
    print("示例 6: 备份数据")
    print("="*60)

    config = MilvusConfig(uri="./milvus_demo.db")
    client = create_client(config)

    collection_name = "backup_demo"
    backup_file = "milvus_backup.json"

    # 1. 创建并插入数据
    create_simple_collection(client, collection_name, dimension=16)

    data = [
        {
            "vector": [random.random() for _ in range(16)],
            "text": f"Document {i}",
            "timestamp": f"2024-01-{i+1:02d}"
        }
        for i in range(10)
    ]

    insert_data(client, collection_name, data)
    print(f"\n创建了 {len(data)} 条记录")

    # 2. 备份数据
    print(f"\n备份数据到 {backup_file}...")
    backup_result = backup_collection_data(
        client,
        collection_name,
        output_file=backup_file
    )
    print(f"结果: {backup_result.message}")

    if backup_result.success:
        print(f"备份文件: {backup_result.data.get('file', '')}")
        print(f"记录数: {backup_result.data.get('count', 0)}")

    # 3. 清理
    print(f"\n清理: 删除 collection...")
    drop_collection(client, collection_name)


def example_memory_management():
    """示例 7: 内存管理"""
    print("\n" + "="*60)
    print("示例 7: 内存管理 (加载/释放 collection)")
    print("="*60)

    config = MilvusConfig(uri="./milvus_demo.db")
    client = create_client(config)

    collection_name = "memory_demo"

    # 1. 创建 collection
    result = create_simple_collection(client, collection_name, dimension=32)

    if not result.success:
        print(f"❌ {result.message}")
        return

    print(f"✅ Collection 已创建并自动加载到内存")

    # 2. 释放内存
    print("\n从内存中释放 collection...")
    release_result = release_collection(client, collection_name)
    print(f"结果: {release_result.message}")

    # 3. 重新加载
    print("\n重新加载 collection 到内存...")
    load_result = load_collection(client, collection_name)
    print(f"结果: {load_result.message}")

    # 4. 清理
    print(f"\n清理: 删除 collection...")
    drop_collection(client, collection_name)


def main():
    """运行所有示例"""
    print("\n" + "="*60)
    print("Milvus 工具使用示例集合")
    print("="*60)

    try:
        example_basic_operations()
        example_create_and_manage_collection()
        example_insert_and_query()
        example_vector_search()
        example_delete_operations()
        example_backup_and_restore()
        example_memory_management()

        print("\n" + "="*60)
        print("✅ 所有示例运行完成")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
