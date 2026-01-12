from pymilvus import MilvusClient, DataType
# 或者使用 Collection 对象
from pymilvus import Collection
from local_llm_direct import DirectOllamaLLM

def check_collection_status(milvus_client, collection_name):
    """检查集合和索引状态"""
    # 检查集合是否存在
    has_col = milvus_client.has_collection(collection_name)
    print(f"集合存在: {has_col}")
    
    # 获取集合详情
    try:
        collection_info = milvus_client.describe_collection(collection_name)
        print(f"集合详情: {collection_info}")
    except Exception as e:
        print(f"获取集合详情失败: {e}")
    
    # 检查索引
    try:
        indexes = milvus_client.list_indexes(collection_name)
        print(f"索引列表: {indexes}")
        
        if indexes:
            for idx in indexes:
                print(f"索引详情: {idx}")
                # 获取特定索引的详细信息
                index_info = milvus_client.describe_index(
                    collection_name=collection_name,
                    index_name=idx
                )
                print(f"索引 {idx} 详情: {index_info}")
        else:
            print("⚠️ 警告: 没有找到任何索引")
            
    except Exception as e:
        print(f"获取索引失败: {e}")


def main():
    milvus_client = MilvusClient(uri="/Users/ruijie/Documents/git/my_silicon_brain/milvus_demo.db")
    collection_name = "finance_knowledge"
    check_collection_status(milvus_client, collection_name)
    # 1. 定义 Schema
    # schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
    # schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    # schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
    # schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=512) # 示例字段

    # # 2. 定义 Index（在这里指定 metric_type）
    # # 注意：本地模式只支持 FLAT, IVF_FLAT, AUTOINDEX, 不支持HNSW模式
    # index_params = MilvusClient.prepare_index_params()
    # index_params.add_index(
    #     field_name="vector",
    #     index_type="HNSW",  # 本地模式使用 FLAT 索引
    #     index_name="vector_index", # Name of the index to create
    #     metric_type="COSINE",   # IP： inner product，这个选择会影响到索引的构建
    #     params={
    #         "M": 64, # Maximum number of neighbors each node can connect to in the graph
    #         "efConstruction": 100 # Number of candidate neighbors considered for connection during index construction
    #         } # Index building params
    # )

    # # 3. 创建集合
    # milvus_client.create_collection(
    #     collection_name=self.collection_name,
    #     schema=schema,             # 必须传入 schema
    #     index_params=index_params, # 推荐传入 index
    #     consistency_level="Bounded"
    # )
    query = '今年国庆节消费怎么样？'
    llm = DirectOllamaLLM(model='qwen3:0.6b', embed_model='qwen3-embedding:0.6b')
    query_embedding = llm.embed(query)

    # 在创建集合后，必须显式加载集合到内存，HNSW和IVF_FLAT需要加载到内存里面才能使用，但是FLAT方法不用
    milvus_client.load_collection(collection_name=collection_name)

    search_params = {
    "params": {
        "ef": 10, # Number of neighbors to consider during the search
    }
    }

    res = milvus_client.search(
        collection_name=collection_name, # Collection name
        anns_field="vector", # Vector field name
        data=[query_embedding],  # Query vector
        limit=10,  # TopK results to return
        search_params=search_params,
        output_fields=["text", "source", "element_type", "metadata", "timestamp"]
    )

    print(res)

if __name__ == "__main__":
    main()
