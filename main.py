#!/usr/bin/env python3
"""
投资研究支持系统主启动脚本
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

from investment_research_system import InvestmentResearchSystem
from knowledge_base import start_document_monitor
from local_llm_config import LocalLLMConfig


def print_banner():
    """打印启动横幅"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    投资研究支持系统 v1.0                        ║
║                  Investment Research System                  ║
║                                                              ║
║  功能特性:                                                    ║
║  • 智能知识库管理 (Milvus + 实时监控)                          ║
║  • MCP网络搜索工具                                            ║
║  • 本地LLM模型支持 (Qwen/DeepSeek)                           ║ 
║  • 板块分析和投资机会识别                                      ║
║  • 交易复盘和策略检查                                          ║
║  • 深度思考编排 (LangChain)                                   ║
╚══════════════════════════════════════════════════════════════╝
    """)


def check_environment():
    """检查环境配置"""
    print("🔍 检查环境配置...")
    
    # 检查.env文件
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env文件不存在，创建模板...")
        create_env_template()
    
    # 检查必要目录
    directories = ["src", "documents", "data"]
    for dir_name in directories:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(exist_ok=True)
            print(f"📁 创建目录: {dir_name}")
    
    # 检查本地LLM配置
    llm_config = LocalLLMConfig()
    available = llm_config.list_available_models()
    
    if available.get("ollama", {}).get("status") == "available":
        models = available["ollama"].get("models", [])
        print(f"✅ Ollama可用，已安装模型: {models}")
    else:
        print("⚠️  Ollama不可用，将使用Claude API")
        print("💡 要使用本地模型，请运行: python src/local_llm_config.py")
    
    print("✅ 环境检查完成\n")


def create_env_template():
    """创建.env模板文件"""
    env_template = """
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Milvus配置
MILVUS_URI=./milvus_demo.db

# Ollama配置 (可选，用于本地LLM)
OLLAMA_BASE_URL=http://localhost:11434

# 其他配置
DOCUMENTS_DIR=documents
MAX_TOKENS=4000
"""
    
    with open(".env", "w") as f:
        f.write(env_template.strip())
    
    print("📝 已创建.env模板文件，请填入您的API密钥")


async def start_knowledge_monitor_async():
    """异步启动知识库监控"""
    try:
        from knowledge_base import start_document_monitor
        kb = start_document_monitor("documents")
        return kb
    except Exception as e:
        print(f"⚠️  启动知识库监控失败: {e}")
        return None


def show_help():
    """显示帮助信息"""
    help_text = """
使用方法:
  python main.py [选项]

选项:
  --help, -h           显示此帮助信息
  --setup             运行环境设置向导
  --knowledge-only    仅启动知识库监控
  --check-llm         检查本地LLM配置
  --interactive       启动交互式界面 (默认)

示例:
  python main.py                    # 启动完整系统
  python main.py --setup           # 运行设置向导
  python main.py --knowledge-only  # 仅监控documents文件夹
  python main.py --check-llm       # 检查本地模型
"""
    print(help_text)


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="投资研究支持系统")
    parser.add_argument("--setup", action="store_true", help="运行环境设置向导")
    parser.add_argument("--knowledge-only", action="store_true", help="仅启动知识库监控")
    parser.add_argument("--check-llm", action="store_true", help="检查本地LLM配置")
    parser.add_argument("--interactive", action="store_true", default=True, help="启动交互式界面")
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.setup:
        print("🔧 运行环境设置向导...")
        check_environment()
        llm_config = LocalLLMConfig()
        print(llm_config.setup_instructions())
        return
    
    if args.check_llm:
        print("🔍 检查本地LLM配置...")
        llm_config = LocalLLMConfig()
        from local_llm_config import main as check_llm_main
        check_llm_main()
        return
    
    if args.knowledge_only:
        print("📚 启动知识库监控模式...")
        try:
            kb = start_document_monitor("documents")
            print("✅ 知识库监控已启动，按Ctrl+C停止")
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 知识库监控已停止")
        return
    
    # 默认启动完整系统
    check_environment()
    
    print("🚀 启动投资研究支持系统...")
    
    try:
        # 启动知识库监控（后台任务）
        print("📚 初始化知识库...")
        
        # 启动主系统
        print("🤖 初始化AI助手...")
        system = InvestmentResearchSystem()
        await system.initialize()
        
        print("✅ 系统启动完成！")
        print("💡 提示：输入 'help' 查看使用说明\n")
        
        # 启动交互循环
        await system.chat_loop()
        
    except KeyboardInterrupt:
        print("\n👋 感谢使用投资研究支持系统！")
    except Exception as e:
        print(f"❌ 系统启动失败: {e}")
        print("💡 请尝试运行 'python main.py --setup' 检查配置")
    finally:
        if 'system' in locals():
            await system.cleanup()


if __name__ == "__main__":
    # 确保在正确的目录下运行
    os.chdir(Path(__file__).parent)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 再见！")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)