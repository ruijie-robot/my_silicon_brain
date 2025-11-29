#!/usr/bin/env python3
"""
投资研究支持系统演示脚本
"""

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from knowledge_base import KnowledgeBase
from local_llm_config import LocalLLMConfig


async def demo_knowledge_base():
    """演示知识库功能"""
    print("\n=== 知识库功能演示 ===")
    
    kb = KnowledgeBase()
    
    # 扫描documents目录
    documents_dir = "documents"
    if Path(documents_dir).exists():
        print(f"📁 扫描目录: {documents_dir}")
        kb.scan_directory(documents_dir)
    else:
        print(f"⚠️  目录 {documents_dir} 不存在，创建示例文档...")
    
    # 测试搜索
    print("\n🔍 测试知识库搜索...")
    search_queries = [
        "价值投资",
        "风险管理",
        "技术分析",
        "投资策略"
    ]
    
    for query in search_queries:
        print(f"\n查询: {query}")
        results = kb.search(query, limit=2)
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['text'][:100]}... (相关度: {result['score']:.3f})")



def demo_local_llm_direct():
    """演示本地LLM功能 (直接函数调用方式)"""
    print("\n=== 本地LLM功能演示 (直接调用) ===")
    
    try:
        from local_llm_direct import DirectOllamaLLM
        
        direct_llm = DirectOllamaLLM()
        
        # 获取可用模型
        models = direct_llm.list_models()
        print(f"📋 可用模型: {models}")
        
        if not models:
            print("⚠️  没有找到可用的Ollama模型")
            print("💡 请先安装模型: ollama pull qwen2.5:latest")
            return
        
        # 选择第一个可用模型
        model_name = models[1]
        embed_model_name = models[0]
        print(f"\n🤖 使用模型: {model_name}, 使用的embed模型: {embed_model_name}")
        
        # 测试问题
        test_queries = [
            {
                "prompt": "简述中国股市的主要特点",
                "system": "你是专业的金融分析师，请用简洁的语言回答。"
            },
            {
                "prompt": "解释价值投资的核心理念", 
                "system": "你是巴菲特的学生，请传授价值投资的精髓。"
            },
            {
                "prompt": "新能源汽车板块有哪些投资机会？",
                "system": "你是新能源行业专家，请分析投资前景。"
            }
        ]
        
        # 逐个测试
        for i, query_info in enumerate(test_queries, 1):
            print(f"\n📝 问题 {i}: {query_info['prompt']}")
            print(f"🎭 角色: {query_info['system']}")
            
            response = direct_llm.simple_chat(
                model=model_name,
                prompt=query_info['prompt'],
                system_prompt=query_info['system']
            )
            
            print(f"💬 回答: {response[:300]}...")
            
            if i < len(test_queries):
                print("-" * 50)
        
        # 演示多轮对话
        print(f"\n🔄 演示多轮对话功能:")
        print("=" * 50)
        
        conversation_messages = [
            {'role': 'system', 'content': '你是专业的股票投资顾问'},
            {'role': 'user', 'content': '我想了解茅台股票'},
        ]
        
        print("用户: 我想了解茅台股票")
        
        response1 = direct_llm.chat(model_name, conversation_messages)
        print(f"助手: {response1[:200]}...")
        
        # 添加AI回复到对话历史
        conversation_messages.append({'role': 'assistant', 'content': response1})
        conversation_messages.append({'role': 'user', 'content': '现在价格合适买入吗？'})
        
        print("\n用户: 现在价格合适买入吗？")
        
        response2 = direct_llm.chat(model_name, conversation_messages)
        print(f"助手: {response2[:200]}...")
        
        # 演示流式输出
        print(f"\n🌊 演示流式输出:")
        print("=" * 50)
        print("问题: 分析一下比亚迪的投资价值")
        print("回答: ", end="", flush=True)
        
        for chunk in direct_llm.stream_chat(model_name, "分析一下比亚迪的投资价值，包括技术优势和市场前景"):
            print(chunk, end="", flush=True)
        print()  # 换行
            
    except ImportError:
        print("❌ 需要安装ollama-python包")
        print("💡 运行: pip install ollama")
    except Exception as e:
        print(f"❌ 直接调用演示失败: {e}")
        print("💡 请确保Ollama服务正在运行: ollama serve")


async def demo_web_search():
    """演示网络搜索功能"""
    print("\n=== 网络搜索功能演示 ===")
    
    try:
        from mcp_web_search import WebSearcher
        
        searcher = WebSearcher()
        
        # 测试搜索功能
        print("🌐 测试DuckDuckGo搜索...")
        results = searcher.search_duckduckgo("新能源汽车 股票", 3)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['title']}")
            print(f"   URL: {result['url']}")
            print(f"   摘要: {result['snippet'][:100]}...")
        
        # 测试金融新闻
        print("\n📰 测试金融新闻获取...")
        news = searcher.search_finance_news("新能源")
        
        for i, item in enumerate(news[:3], 1):
            print(f"\n{i}. {item['title']}")
            print(f"   来源: {item['source']}")
            print(f"   摘要: {item['summary'][:100]}...")
            
    except Exception as e:
        print(f"❌ 网络搜索演示失败: {e}")
        print("💡 请检查网络连接和依赖包")


def demo_file_structure():
    """显示项目文件结构"""
    print("\n=== 项目文件结构 ===")
    
    def print_tree(directory, prefix="", level=0):
        if level > 2:  # 限制显示层级
            return
            
        path = Path(directory)
        if not path.exists():
            return
            
        items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
        
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item.name}")
            
            if item.is_dir() and not item.name.startswith('.'):
                extension = "    " if is_last else "│   "
                print_tree(item, prefix + extension, level + 1)
    
    print("📁 项目目录结构:")
    print_tree(".")


async def main():
    """主演示函数"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                  投资研究支持系统演示                          ║
║                     System Demo                             ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 显示项目结构
    # demo_file_structure()
    
    # 演示各个功能模块
    try:
        await demo_knowledge_base()
    except Exception as e:
        print(f"❌ 知识库演示失败: {e}")
    
    # try:
    #     demo_local_llm()
    # except Exception as e:
    #     print(f"❌ 本地LLM演示失败: {e}")
    
    # 演示直接调用版本: Ruijie: done
    # try:
    #     demo_local_llm_direct()
    # except Exception as e:
    #     print(f"❌ 本地LLM直接调用演示失败: {e}")
    
    # try:
    #     await demo_web_search()
    # except Exception as e:
    #     print(f"❌ 网络搜索演示失败: {e}")


if __name__ == "__main__":
    asyncio.run(main())