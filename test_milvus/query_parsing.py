from pymilvus import MilvusClient, DataType
# 或者使用 Collection 对象
from pymilvus import Collection
from local_llm_direct import DirectOllamaLLM


def main():
    user_query = '今年国庆节消费怎么样？'
    llm = DirectOllamaLLM(model='qwen3:0.6b', embed_model='qwen3-embedding:0.6b')
    prompt = f"以下prompt是用户用于对金融知识库进行查询的语句: \
            {user_query} \
            请重新写prompt来优化知识库的搜索结果，搜索结果需要符合以下要求: \
            - 澄清模糊的短语 \
            - 在适当的时候使用金融术语 \
            - 添加能提高匹配文档命中率的同义词 \
            - 删除不必要的干扰信息"
    results = llm.simple_chat(prompt)
    print(results)

if __name__ == "__main__":
    main()
