"""
使用ollama-python直接调用的版本示例
需要安装: pip install ollama
"""

import ollama
from typing import Optional, List, Dict, Any
from ollama import chat, embeddings
from ollama import ChatResponse

class DirectOllamaLLM:
    def __init__(self, model: str = 'qwen3:0.6b', embed_model: str = 'qwen3-embedding:0.6b'):
        self.model = model
        self.embed_model = embed_model
        self.client = ollama.Client()
    
    def list_ollama_models(self) -> List[str]:
        """列出可用模型"""
        try:
            models = self.client.list()
            return [model.model for model in models['models']]
        except Exception as e:
            print(f"获取模型列表失败: {e}")
            return []
    
    
    def simple_chat(self, prompt: str, 
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
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    'temperature': 0.8,
                    'num_predict': 2000
                }
            )
            return response['message']['content']
        except Exception as e:
            return f"调用失败: {str(e)}"
    
    def stream_chat(self, prompt: str):
        """流式聊天"""
        try:
            stream = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                stream=True,
            )
            
            for chunk in stream:
                yield chunk['message']['content']
                
        except Exception as e:
            yield f"错误: {str(e)}"

    def embed(self, text: str) -> list:
        """
        使用指定模型对一句话(text)生成embedding向量。

        :param model: embedding模型名称 (如 'qwen3-embedding:0.6b')
        :param text: 要编码的文本
        :return: embedding向量（list[float]）
        """
        try:
            result = self.client.embed(
                model=self.embed_model,
                input=text
            )
            # Ollama返回 dict，字段为 'embeddings'，为list (通常长度1)
            embeddings = result.get('embeddings', [])
            if embeddings:
                return embeddings[0]
            else:
                return []
        except Exception as e:
            # 可选: 返回空或者异常信息
            return []



# 使用示例
if __name__ == "__main__":
    llm = DirectOllamaLLM()
    
    # 列出模型
    models = llm.list_ollama_models()
    print(f"可用模型: {models}")
        
    # 'qwen3-embedding:0.6b'
    embeddings = llm.embed(
    text='The quick brown fox jumps over the lazy dog.'
    )
    print(embeddings)  # vector length

    # 简单对话 'qwen3:0.6b'
    response = llm.simple_chat(
        prompt="简述中国股市特点",
        system_prompt="你是专业的金融分析师"
    )
    print(f"回答: {response}")
    
    # 流式对话
    print("\n流式输出:")
    for chunk in llm.stream_chat("分析新能源汽车板块"):
        print(chunk, end='', flush=True)

    print("Finished")