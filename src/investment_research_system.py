import asyncio
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextlib import AsyncExitStack

from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage

from knowledge_base import KnowledgeBase

load_dotenv()


class InvestmentResearchSystem:
    def __init__(self):
        self.anthropic = Anthropic()
        self.knowledge_base = KnowledgeBase()
        self.memory = ConversationBufferWindowMemory(k=10, return_messages=True)
        
        # MCP相关
        self.sessions: List[ClientSession] = []
        self.exit_stack = AsyncExitStack()
        self.available_tools = []
        self.tool_to_session = {}
        
        # 系统配置
        self.model = "claude-3-7-sonnet-20250219"
        self.max_tokens = 4000
        
        # 投资研究提示词
        self.system_prompts = {
            "sector_analysis": """
你是一位专业的基金经理助手，专门负责板块分析。
当用户询问某个板块时，你需要：
1. 分析该板块的基本情况和发展趋势
2. 识别板块内的优质公司和投资机会
3. 结合用户的投资经验，推荐相对熟悉的标的
4. 提供风险提示和投资建议
5. 使用搜索工具获取最新的市场信息和新闻

请保持专业、客观的分析态度，结合基本面和技术面进行分析。
""",
            "review_analysis": """
你是一位专业的交易复盘分析师。
当用户提交交易复盘或投资计划时，你需要：
1. 仔细分析用户的交易记录和思考过程
2. 检查是否遗漏了重要的投资机会
3. 识别可能违背投资原则的操作
4. 提供建设性的改进建议
5. 总结经验教训和优化建议

请基于用户的历史数据和知识库信息进行深度分析。
"""
        }
    
    async def initialize(self):
        """初始化系统"""
        await self.connect_to_mcp_servers()
        self._start_knowledge_monitor()
    
    async def connect_to_mcp_servers(self):
        """连接到MCP服务器"""
        servers = {
            "web_search": {
                "command": "python",
                "args": ["src/mcp_web_search.py"]
            },
            "research": {
                "command": "python", 
                "args": ["test_mcp/research_server.py"]
            }
        }
        
        for server_name, server_config in servers.items():
            try:
                server_params = StdioServerParameters(**server_config)
                stdio_transport = await self.exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                read, write = stdio_transport
                session = await self.exit_stack.enter_async_context(
                    ClientSession(read, write)
                )
                await session.initialize()
                self.sessions.append(session)
                
                # 获取工具列表
                response = await session.list_tools()
                tools = response.tools
                print(f"Connected to {server_name} with tools: {[t.name for t in tools]}")
                
                for tool in tools:
                    self.tool_to_session[tool.name] = session
                    self.available_tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    })
                    
            except Exception as e:
                print(f"Failed to connect to {server_name}: {e}")
    
    def _start_knowledge_monitor(self):
        """启动知识库监控（在后台运行）"""
        # 这里可以启动文档监控，在实际应用中可能需要在单独的进程中运行
        pass
    
    def _build_context(self, query: str, max_results: int = 3) -> str:
        """从知识库构建上下文"""
        search_results = self.knowledge_base.search(query, max_results)
        
        if not search_results:
            return ""
        
        context_parts = []
        for result in search_results:
            context_parts.append(f"""
来源: {result['source']}
内容: {result['text']}
相关度: {result['score']:.3f}
""")
        
        return f"""
=== 知识库相关信息 ===
{chr(10).join(context_parts)}
=== 知识库信息结束 ===
"""
    
    def _get_conversation_history(self) -> str:
        """获取对话历史"""
        messages = self.memory.chat_memory.messages
        if not messages:
            return ""
        
        history_parts = []
        for msg in messages[-6:]:  # 只取最近6条消息
            if isinstance(msg, HumanMessage):
                history_parts.append(f"用户: {msg.content}")
            elif isinstance(msg, AIMessage):
                history_parts.append(f"助手: {msg.content}")
        
        return f"""
=== 对话历史 ===
{chr(10).join(history_parts)}
=== 对话历史结束 ===
"""
    
    async def analyze_sector(self, sector_query: str) -> str:
        """板块分析"""
        # 构建上下文
        context = self._build_context(f"{sector_query} 板块 投资")
        history = self._get_conversation_history()
        
        # 构建完整提示
        full_prompt = f"""
{self.system_prompts['sector_analysis']}

{context}

{history}

用户查询: {sector_query}

请基于知识库信息和最新市场数据，提供专业的板块分析。
"""
        
        return await self._process_with_tools(full_prompt)
    
    async def review_trading_plan(self, plan_content: str) -> str:
        """交易复盘分析"""
        # 构建相关的上下文
        context = self._build_context(f"交易原则 投资策略 风险管理")
        history = self._get_conversation_history()
        
        full_prompt = f"""
{self.system_prompts['review_analysis']}

{context}

{history}

用户的交易复盘/计划:
{plan_content}

请仔细分析这份复盘或计划，检查是否有遗漏的机会或违背投资原则的地方。
"""
        
        return await self._process_with_tools(full_prompt)
    
    async def general_query(self, query: str) -> str:
        """通用查询处理"""
        context = self._build_context(query)
        history = self._get_conversation_history()
        
        full_prompt = f"""
你是一位专业的基金经理助手，擅长投资研究和分析。
请根据用户的问题，结合知识库信息和实时数据，提供专业的回答。

{context}

{history}

用户问题: {query}
"""
        
        return await self._process_with_tools(full_prompt)
    
    async def _process_with_tools(self, prompt: str) -> str:
        """使用工具处理查询"""
        messages = [{'role': 'user', 'content': prompt}]
        
        try:
            response = self.anthropic.messages.create(
                max_tokens=self.max_tokens,
                model=self.model,
                tools=self.available_tools,
                messages=messages
            )
            
            # 处理响应和工具调用
            final_response = await self._handle_response_with_tools(response, messages)
            
            # 保存到记忆
            self.memory.chat_memory.add_user_message(prompt)
            self.memory.chat_memory.add_ai_message(final_response)
            
            return final_response
            
        except Exception as e:
            error_msg = f"处理查询时出错: {str(e)}"
            print(error_msg)
            return error_msg
    
    async def _handle_response_with_tools(self, response, messages: List[Dict]) -> str:
        """处理带工具调用的响应"""
        while True:
            assistant_content = []
            text_response = ""
            
            for content in response.content:
                if content.type == 'text':
                    text_response = content.text
                    assistant_content.append(content)
                    if len(response.content) == 1:
                        return text_response
                        
                elif content.type == 'tool_use':
                    assistant_content.append(content)
                    messages.append({'role': 'assistant', 'content': assistant_content})
                    
                    # 执行工具调用
                    tool_result = await self._call_tool(content.name, content.input)
                    
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": tool_result
                        }]
                    })
                    
                    # 获取下一个响应
                    response = self.anthropic.messages.create(
                        max_tokens=self.max_tokens,
                        model=self.model,
                        tools=self.available_tools,
                        messages=messages
                    )
                    
                    if (len(response.content) == 1 and 
                        response.content[0].type == "text"):
                        return response.content[0].text
                    
                    break  # 继续下一轮循环
    
    async def _call_tool(self, tool_name: str, tool_args: Dict) -> str:
        """调用MCP工具"""
        try:
            if tool_name in self.tool_to_session:
                session = self.tool_to_session[tool_name]
                result = await session.call_tool(tool_name, arguments=tool_args)
                return str(result.content)
            else:
                return f"工具 {tool_name} 不可用"
        except Exception as e:
            return f"调用工具 {tool_name} 时出错: {str(e)}"
    
    async def chat_loop(self):
        """交互式对话循环"""
        print("=== 投研支持系统启动 ===")
        print("命令:")
        print("- 'sector <板块名>' - 分析特定板块")
        print("- 'review <内容>' - 交易复盘分析") 
        print("- 'quit' - 退出系统")
        print("- 其他问题直接输入")
        print("========================")
        
        while True:
            try:
                user_input = input("\n请输入您的问题: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                
                if user_input.startswith('sector '):
                    sector = user_input[7:]
                    response = await self.analyze_sector(sector)
                elif user_input.startswith('review '):
                    content = user_input[7:]
                    response = await self.review_trading_plan(content)
                else:
                    response = await self.general_query(user_input)
                
                print(f"\n助手: {response}")
                print("\n" + "="*50)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"错误: {str(e)}")
    
    async def cleanup(self):
        """清理资源"""
        await self.exit_stack.aclose()


async def main():
    system = InvestmentResearchSystem()
    try:
        await system.initialize()
        await system.chat_loop()
    finally:
        await system.cleanup()


if __name__ == "__main__":
    asyncio.run(main())