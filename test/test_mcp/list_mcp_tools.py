#!/usr/bin/env python3
"""
查看 MCP 服务器提供的 tools 的脚本
"""
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack


async def list_tools_from_server(server_name: str, command: str, args: list):
    """连接到 MCP 服务器并列出所有 tools"""
    print(f"\n{'='*60}")
    print(f"连接到服务器: {server_name}")
    print(f"命令: {command} {' '.join(args)}")
    print(f"{'='*60}\n")
    
    try:
        server_params = StdioServerParameters(
            command=command,
            args=args
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # 初始化连接
                await session.initialize()
                
                # 列出所有 tools
                response = await session.list_tools()
                tools = response.tools
                
                print(f"✅ 找到 {len(tools)} 个工具:\n")
                
                for i, tool in enumerate(tools, 1):
                    print(f"{i}. {tool.name}")
                    print(f"   描述: {tool.description}")
                    print(f"   参数架构:")
                    print(f"   {json.dumps(tool.inputSchema, indent=6, ensure_ascii=False)}")
                    print()
                
                return tools
                
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        import traceback
        traceback.print_exc()
        return []


async def main():
    """主函数"""
    import sys
    
    # 如果提供了服务器配置，使用它
    if len(sys.argv) > 1:
        server_name = sys.argv[1]
        command = sys.argv[2] if len(sys.argv) > 2 else "python"
        args = sys.argv[3:] if len(sys.argv) > 3 else []
        
        await list_tools_from_server(server_name, command, args)
    else:
        # 默认查看 web_search 服务器
        print("使用默认配置查看 web_search 服务器")
        print("用法: python list_mcp_tools.py [server_name] [command] [args...]")
        print("示例: python list_mcp_tools.py web_search python src/mcp_web_search.py\n")
        
        await list_tools_from_server(
            "web_search",
            "python",
            ["src/mcp_web_search.py"]
        )


if __name__ == "__main__":
    asyncio.run(main())

