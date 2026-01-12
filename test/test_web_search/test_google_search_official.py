import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import sys
import os
from pathlib import Path

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
import json

load_dotenv()

# --- 从环境变量获取配置 ---
# 确保在运行代码前，您的环境中已设置 GOOGLE_API_KEY 和 GOOGLE_SEARCH_CX
# API_KEY = os.environ.get('GOOGLE_API_KEY')
# CX_ID = os.environ.get('GOOGLE_SEARCH_CX')
API_KEY = os.getenv("GOOGLE_API_KEY")
CX_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID") 

def custom_google_search(query: str, num_results: int = 5) -> list:
    """
    使用 Google Custom Search API 执行搜索。

    Args:
        query: 要搜索的关键词。
        num_results: 希望返回的结果数量 (最大为 10)。

    Returns:
        包含搜索结果字典的列表，每个字典包含 'title', 'link', 'snippet'。
        如果失败或无结果，返回空列表。
    """
    if not API_KEY or not CX_ID:
        print("错误：请先设置 GOOGLE_API_KEY 和 GOOGLE_SEARCH_CX 环境变量。")
        return []

    try:
        # 1. 构建服务对象：指定 API 名称和版本
        service = build("customsearch", "v1", developerKey=API_KEY)
        
        # 2. 执行搜索请求
        # q: 搜索关键词, cx: 搜索引擎ID, num: 返回结果数
        res = service.cse().list(
            q=query,
            cx=CX_ID,
            num=min(num_results, 10) # 官方 API 限制最大返回 10 个结果
        ).execute()
        
        # 3. 提取结果
        results = []
        if 'items' in res:
            for item in res['items']:
                results.append({
                    'title': item.get('title'),
                    'link': item.get('link'),
                    'snippet': item.get('snippet')
                })
        
        return results

    except HttpError as e:
        print(f"API 请求失败 (HTTP 错误): {e}")
        # 检查是否是配额不足或 API Key 无效
        return []
    except Exception as e:
        print(f"发生未知错误: {e}")
        return []

# --- 运行示例 ---
# search_query = "Agent Web 搜索工具集成最佳实践 2024"
search_query = "特斯拉股价"
print(f"--- 正在执行搜索: {search_query} ---")
search_results = custom_google_search(search_query, num_results=3)

if search_results:
    for i, item in enumerate(search_results):
        print(f"\n{i+1}. **{item['title']}**")
        print(f"   URL: {item['link']}")
        print(f"   摘要: {item['snippet']}")
else:
    print("未找到任何搜索结果。请检查您的 API Key 或 CX ID 是否有效。")
