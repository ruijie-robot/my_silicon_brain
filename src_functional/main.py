#!/usr/bin/env python3
"""
函数式编程版本的投资研究支持系统演示脚本
使用纯函数和函数组合实现
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Tuple, Callable
from functools import partial

sys.path.append(str(Path(__file__).parent))

from knowledge_base import (
    create_kb_context,
    scan_and_add_directory,
    search_kb,
    load_file_hashes,
    save_file_hashes
)


# ============================================================================
# 纯函数 - 格式化输出
# ============================================================================

def format_search_result(index: int, result, max_length: int = 100) -> str:
    """格式化搜索结果 - 纯函数"""
    text = result.text[:max_length] + "..." if len(result.text) > max_length else result.text
    return f"  {index}. {text} (相关度: {result.score:.3f})"


def format_search_results(results: List, max_length: int = 100) -> List[str]:
    """格式化搜索结果列表 - 纯函数"""
    return [
        format_search_result(i, result, max_length)
        for i, result in enumerate(results, 1)
    ]


def format_header(title: str) -> str:
    """格式化标题 - 纯函数"""
    return f"\n=== {title} ==="


def format_query(query: str) -> str:
    """格式化查询 - 纯函数"""
    return f"\n查询: {query}"


# ============================================================================
# 演示函数 - 知识库功能
# ============================================================================

async def demo_knowledge_base() -> None:
    """演示知识库功能"""
    print(format_header("知识库功能演示"))

    # 创建知识库上下文
    client, coll_config, llm_config, proc_config = create_kb_context(
        collection_name="finance_knowledge_HNSW_functional"
    )

    # 加载文件哈希
    file_hashes = load_file_hashes()

    # 扫描文档目录
    documents_dir = "documents"
    if Path(documents_dir).exists():
        print(f"📁 扫描目录: {documents_dir}")
        file_hashes = scan_and_add_directory(
            client,
            coll_config.collection_name,
            documents_dir,
            llm_config,
            proc_config,
            file_hashes
        )

        # 保存哈希
        save_file_hashes(file_hashes)
    else:
        print(f"⚠️  目录 {documents_dir} 不存在")

    # 测试搜索
    print("\n🔍 测试知识库搜索...")
    search_queries = [
        "今年国庆节消费怎么样？",
        "4月关税对中国股市的冲击?",
        "黄金未来走势？",
        "特斯拉未来走势？"
    ]

    # 创建预配置的搜索函数
    search_fn = partial(
        search_kb,
        client=client,
        collection_name=coll_config.collection_name,
        llm_config=llm_config,
        limit=2
    )

    # 执行搜索并显示结果
    for query in search_queries:
        print(format_query(query))
        results = search_fn(query=query)
        formatted_results = format_search_results(results)

        for formatted_result in formatted_results:
            print(formatted_result)


# ============================================================================
# 演示函数 - 本地 LLM 功能
# ============================================================================

async def demo_local_llm_direct() -> None:
    """演示本地 LLM 功能（直接函数调用方式）"""
    print(format_header("本地LLM功能演示 (函数式)"))

    try:
        from local_llm_direct import (
            create_config,
            list_models,
            simple_chat,
            stream_chat,
            chat_with_history,
            create_user_message,
            create_system_message,
            create_assistant_message
        )

        # 列出可用模型
        models = list_models()
        print(f"📋 可用模型: {models}")

        if not models:
            print("⚠️  没有找到可用的Ollama模型")
            print("💡 请先安装模型: ollama pull qwen2.5:latest")
            return

        # 选择模型
        model_name = models[1] if len(models) > 1 else models[0]
        embed_model_name = models[0]
        print(f"\n🤖 使用模型: {model_name}, 嵌入模型: {embed_model_name}")

        # 创建配置
        config = create_config(model=model_name, embed_model=embed_model_name)

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

            result = simple_chat(
                config=config,
                prompt=query_info['prompt'],
                system_prompt=query_info['system']
            )

            if result.is_success:
                print(f"💬 回答: {result.content[:300]}...")
            else:
                print(f"❌ 错误: {result.error}")

            if i < len(test_queries):
                print("-" * 50)

        # 演示多轮对话
        print(f"\n🔄 演示多轮对话功能:")
        print("=" * 50)

        # 构建对话历史
        messages = [
            create_system_message('你是专业的股票投资顾问'),
            create_user_message('我想了解茅台股票')
        ]

        print("用户: 我想了解茅台股票")

        result1 = chat_with_history(config, messages)
        if result1.is_success:
            print(f"助手: {result1.content[:200]}...")

            # 添加回复到历史
            messages.append(create_assistant_message(result1.content))
            messages.append(create_user_message('现在价格合适买入吗？'))

            print("\n用户: 现在价格合适买入吗？")

            result2 = chat_with_history(config, messages)
            if result2.is_success:
                print(f"助手: {result2.content[:200]}...")
            else:
                print(f"❌ 错误: {result2.error}")
        else:
            print(f"❌ 错误: {result1.error}")

        # 演示流式输出
        print(f"\n🌊 演示流式输出:")
        print("=" * 50)
        print("问题: 分析一下比亚迪的投资价值")
        print("回答: ", end="", flush=True)

        for chunk in stream_chat(
            config,
            "分析一下比亚迪的投资价值，包括技术优势和市场前景"
        ):
            print(chunk, end="", flush=True)
        print()

    except ImportError:
        print("❌ 需要安装ollama-python包")
        print("💡 运行: pip install ollama")
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        print("💡 请确保Ollama服务正在运行: ollama serve")


# ============================================================================
# 主函数
# ============================================================================

async def main() -> None:
    """主演示函数"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║              投资研究支持系统演示 (函数式版本)                 ║
║           Investment Research System Demo (Functional)       ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # 演示知识库功能
    try:
        await demo_knowledge_base()
    except Exception as e:
        print(f"❌ 知识库演示失败: {e}")

    # 演示本地 LLM 功能
    # try:
    #     await demo_local_llm_direct()
    # except Exception as e:
    #     print(f"❌ 本地LLM演示失败: {e}")


if __name__ == "__main__":
    asyncio.run(main())
