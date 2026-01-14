#!/usr/bin/env python3
"""
Milvus 工具使用示例
展示常见的数据库操作场景
"""

from milvus_tool import (
    MilvusConfig,
    create_client,
    list_collections, # ok
    print_collection_info, #ok
    create_HNSW_collection, # ok
    drop_collection, # ok
    insert_data, # ok
    query_data, # ok
    search_vectors, # ok, 找最相关的k个值
    delete_data # ？
)
import random



def insert_and_query(config, client, collection_name):
    # 1. 准备数据
    documents = [
        {
            "vector": [random.random() for _ in range(128)],
            "text": "人工智能是计算机科学的一个分支",
            "vector": [random.random() for _ in range(64)],
            "category": "AI",
            "importance": 5
        },
        {
            "vector": [random.random() for _ in range(128)],
            "text": "机器学习是实现人工智能的一种方法",
             "vector": [random.random() for _ in range(64)],
            "category": "ML",
            "importance": 4
        },
        {
            "vector": [random.random() for _ in range(128)],
            "text": "深度学习是机器学习的一个子领域",
             "vector": [random.random() for _ in range(64)],
            "category": "DL",
            "importance": 5
        },
        {
            "vector": [random.random() for _ in range(128)],
            "text": "自然语言处理研究计算机与人类语言的交互",
             "vector": [random.random() for _ in range(64)],
            "category": "NLP",
            "importance": 4
        },
        {
            "vector": [random.random() for _ in range(128)],
            "text": "计算机视觉让计算机能够理解和处理图像",
             "vector": [random.random() for _ in range(64)],
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
        limit=3
    )
    print(f"结果: {query_result.data}")

    # 5. 条件查询
    print("\n查询重要性 >= 5 的记录...")
    filtered_result = query_data(
        client,
        collection_name,
        filter_expr="importance >= 5",
        output_fields=["text", "importance"]
    )
    print(f"结果: {filtered_result.data}")



def vector_search(client, collection_name):
    # 3. 生成查询向量
    query_vector = [random.random() for _ in range(64)]

    # 4. 执行向量搜索
    print("\n执行向量搜索...")
    search_result = search_vectors(
        client,
        collection_name,
        query_vectors=[query_vector],
        limit=3,
        output_fields=["text", "score"]
    )

    print(f"结果: {search_result.data}")

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
    print(f"结果: {search_result.data}")


def delete_operations(client, collection_name):
    # 2. 查询初始状态
    initial_query = query_data(client, collection_name, filter_expr="id >= 0", limit=100)
    print(f"初始记录数: {len(initial_query.data) if initial_query.data else 0}")

    # 3. 按 ID 删除
    if initial_query.data[0]['id']:
        ids_to_delete = [initial_query.data[0]['id'], initial_query.data[1]['id']]
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



def main():
    """运行所有示例"""
    # 1. 创建客户端
    config = MilvusConfig(uri="../milvus_demo.db")
    client = create_client(config)
    collection_name = "test"

    try:
        ### list collections
        collections = list_collections(client)
        print(f"\n当前 collections: {collections}")


        print_collection_info(client, collection_name)

        ### create collection
        result = create_HNSW_collection(
                client,
                collection_name="test",
                dimension=64
        )
        print(f"结果: {result.message}")

        insert_and_query(config, client, collection_name)


        vector_search(client, collection_name)
        delete_operations(client, collection_name)

        # 6. 清理
        print(f"\n清理: 删除 collection...")
        drop_collection(client, collection_name)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
