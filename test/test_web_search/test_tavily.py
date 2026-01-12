from tavily import TavilyClient
import sys
import os
from pathlib import Path

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
import json

load_dotenv()

API_KEY = os.getenv("TAVILY_API_KEY")

# 实例化客户端（会自动读取环境变量中的 API Key）
tavily = TavilyClient(api_key=API_KEY)

# 定义您的搜索查询
# query = "特斯拉（Tesla, TSLA）股票最近的动态、新闻和分析"
query = "中国最大的城市是哪里？"

# 执行搜索
try:
    response = tavily.search(
        query=query, 
        max_results=5,      # 我们想要返回 5 个最相关的来源
        include_answer=True,  # 让 Tavily 生成一个总结性的答案
        search_depth="advanced" # 使用更深入的搜索以获取最新的高质量信息
    )

    print(f"--- 针对查询 '{query}' 的 Tavily 总结 ---")
    
    # 打印 Tavily 生成的总结性答案
    if response['answer']:
        print(response['answer'])
    else:
        print("未生成总结性答案，请查看来源链接。")

    print("\n--- 关键信息来源 ---")
    # 遍历并打印搜索结果的详细来源
    for i, result in enumerate(response['results']):
        print(f"[{i+1}] 标题: {result['title']}")
        print(f"    链接: {result['url']}")
        # 打印来源的摘要（Snippet）以快速了解内容
        print(f"    摘要: {result['content'][:150]}...") # 仅显示前150字符
        print("-" * 30)

except Exception as e:
    print(f"调用 Tavily API 时发生错误: {e}")