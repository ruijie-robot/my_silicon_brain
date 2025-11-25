"""
使用ollama-python直接调用的版本示例
需要安装: pip install ollama
"""

import ollama
from typing import Optional, List, Dict, Any


class DirectOllamaLLM:
    def __init__(self):
        self.client = ollama.Client()
    
    def list_models(self) -> List[str]:
        """列出可用模型"""
        try:
            models = self.client.list()
            return [model.model for model in models['models']]
        except Exception as e:
            print(f"获取模型列表失败: {e}")
            return []
    
    def chat(self, model: str, messages: List[Dict[str, str]], 
             temperature: float = 0.7) -> str:
        """聊天接口"""
        try:
            response = ollama.chat(
                model=model,
                messages=messages,
                options={
                    'temperature': temperature,
                    'num_predict': 2000
                }
            )
            return response['message']['content']
        except Exception as e:
            return f"调用失败: {str(e)}"
    
    def simple_chat(self, model: str, prompt: str, 
                   system_prompt: Optional[str] = None) -> str:
        """简单聊天接口"""
        messages = []
        
        if system_prompt:
            messages.append({
                'role': 'system',
                'content': system_prompt
            })
        
        messages.append({
            'role': 'user', 
            'content': prompt
        })
        
        return self.chat(model, messages)
    
    def stream_chat(self, model: str, prompt: str):
        """流式聊天"""
        try:
            stream = ollama.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                stream=True,
            )
            
            for chunk in stream:
                yield chunk['message']['content']
                
        except Exception as e:
            yield f"错误: {str(e)}"


# 使用示例
if __name__ == "__main__":
    llm = DirectOllamaLLM()
    
    # 列出模型
    models = llm.list_models()
    print(f"可用模型: {models}")
    
    if models:
        model = models[0]
        
        # 简单对话
        response = llm.simple_chat(
            model=model,
            prompt="简述中国股市特点",
            system_prompt="你是专业的金融分析师"
        )
        print(f"回答: {response}")
        
        # 流式对话
        print("\n流式输出:")
        for chunk in llm.stream_chat(model, "分析新能源汽车板块"):
            print(chunk, end='', flush=True)