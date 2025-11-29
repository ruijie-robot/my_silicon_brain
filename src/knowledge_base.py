import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio

from pymilvus import MilvusClient, DataType
from openai import OpenAI
from dotenv import load_dotenv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import unstructured
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.md import partition_md
from local_llm_direct import DirectOllamaLLM

load_dotenv()


class DocumentProcessor:
    def __init__(self):
        # ['qwen3-embedding:0.6b', 'qwen3:0.6b']
        self.llm = DirectOllamaLLM(model='qwen3:0.6b', embed_model='qwen3-embedding:0.6b')

    
    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """处理文档并返回文本块"""
        try:

            # 如果 file_path 以 pdf 结尾，采用partition_pdf方式并配置合适参数，否则用partition
            if file_path.lower().endswith(".pdf"):
                elements = partition_pdf(
                    filename=file_path,
                    languages=["chi_sim", "eng"],  # 支持中文和英文
                    strategy="fast",             # hi_res高精度支持中文, 依赖在线YOLOX模型，而fast不依赖在线
                    extract_images=False,
                    infer_table_structure=True,
                    ocr_mode="entire_page",
                    extract_image_block_to_payload=False,
                )
                print("使用partition_pdf")
            elif file_path.lower().endswith(".md"):
                # 如果以.md结尾，使用partition_md并设置中文、英文语言和合适分块参数
                elements = partition_md(
                    filename=file_path,
                    encoding="utf-8",
                    languages=["chi_sim", "eng"],
                    chunking_strategy="by_title",
                    max_characters=2000,
                    new_after_n_chars=1500,
                )
                print("使用partition_md")
            else:
                print("使用通用的partition")
                elements = partition(filename=file_path)
            chunks = []
            
            for i, element in enumerate(elements):
                if hasattr(element, 'text') and element.text.strip():
                    chunk = {
                        "id": f"{Path(file_path).stem}_{i}",
                        "text": element.text,
                        "source": file_path,
                        "element_type": str(type(element).__name__),
                        "metadata": element.metadata.to_dict() if hasattr(element, 'metadata') else {}
                    }
                    chunks.append(chunk)
            
            return chunks
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []
    
    


class KnowledgeBase:
    def __init__(self, milvus_uri: str = "./milvus_demo.db", collection_name: str = "finance_knowledge"):
        self.milvus_client = MilvusClient(uri=milvus_uri)
        self.collection_name = collection_name
        self.processor = DocumentProcessor()
        self.file_hashes = {}
        self._initialize_collection() #创建collections
        self._load_file_hashes()
    
    def _initialize_collection(self):
        """初始化Milvus集合"""
        if self.milvus_client.has_collection(self.collection_name):
            print(f"Collection {self.collection_name} already exists")
        else:
            # 创建测试向量来确定维度
            test_embedding = self.processor.llm.embed("test")
            if not test_embedding:
                print(f"Error: Failed to create test embedding. Cannot initialize collection {self.collection_name}")
                raise RuntimeError("Failed to create test embedding. Check OpenAI API key and connection.")
            
            dimension = len(test_embedding)
            # 1. 定义 Schema
            schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=512) # 示例字段

            # 2. 定义 Index（在这里指定 metric_type）
            # 注意：本地模式只支持 FLAT, IVF_FLAT, AUTOINDEX, 不支持HNSW模式
            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="FLAT",  # 本地模式使用 FLAT 索引
                metric_type="IP",   # IP： inner product，这个选择会影响到索引的构建
                params={}            # FLAT 索引不需要额外参数
            )

            # 3. 创建集合
            self.milvus_client.create_collection(
                collection_name=self.collection_name,
                schema=schema,             # 必须传入 schema
                index_params=index_params, # 推荐传入 index
                consistency_level="Bounded"
            )

            print(f"创建collection: {self.collection_name} 集合已使用 Schema 和 Index 正确创建。")
    
    def _load_file_hashes(self):
        """加载文件哈希记录"""
        hash_file = "file_hashes.json"
        if os.path.exists(hash_file):
            with open(hash_file, 'r') as f:
                self.file_hashes = json.load(f)
    
    def _save_file_hashes(self):
        """保存文件哈希记录"""
        with open("file_hashes.json", 'w') as f:
            json.dump(self.file_hashes, f, indent=2)
    
    def _get_file_hash(self, file_path: str) -> str:
        """计算文件哈希"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _file_changed(self, file_path: str) -> bool:
        """检查文件是否变更"""
        current_hash = self._get_file_hash(file_path)
        stored_hash = self.file_hashes.get(file_path)
        return current_hash != stored_hash
    
    def add_document(self, file_path: str) -> bool:
        """添加文档到知识库"""
        try:
            if not self._file_changed(file_path):
                print(f"File {file_path} unchanged, skipping")
                return False
            
            # 处理文档
            chunks = self.processor.process_document(file_path)
            if not chunks:
                return False
            
            # 删除旧的文档记录
            self._remove_document(file_path)
            
            # 准备数据用于插入
            data = []
            for chunk in chunks:
                embedding = self.processor.llm.embed(chunk["text"])
                if embedding:
                    data.append({
                        # 不包含 id 字段，因为 schema 设置了 auto_id=True，Milvus 会自动生成 int64 主键
                        "vector": embedding,
                        "text": chunk["text"],
                        "source": chunk["source"],
                        "element_type": chunk["element_type"],
                        "metadata": chunk["metadata"],
                        "timestamp": datetime.now().isoformat(),
                        "chunk_id": chunk["id"]  # 将原来的字符串 ID 保存为动态字段
                    })
            
            # 插入数据
            if data:
                self.milvus_client.insert(collection_name=self.collection_name, data=data)
                
                # 更新哈希记录
                self.file_hashes[file_path] = self._get_file_hash(file_path)
                self._save_file_hashes()
                
                print(f"Added {len(data)} chunks from {file_path}")
                return True
            
            return False
            
        except Exception as e:
            print(f"Error adding document {file_path}: {e}")
            return False
    
    def _remove_document(self, file_path: str):
        """从知识库中移除文档"""
        try:
            # 先检查 collection 是否存在
            if not self.milvus_client.has_collection(self.collection_name):
                print(f"Collection {self.collection_name} does not exist, skipping removal")
                return
            
            # 查找该文件的所有记录
            search_results = self.milvus_client.query(
                collection_name=self.collection_name,
                filter=f'source == "{file_path}"',
                output_fields=["id"]
            )
            
            if search_results:
                ids_to_delete = [result["id"] for result in search_results]
                self.milvus_client.delete(
                    collection_name=self.collection_name,
                    ids=ids_to_delete
                )
                print(f"Removed {len(ids_to_delete)} chunks from {file_path}")
                
        except Exception as e:
            print(f"Error removing document {file_path}: {e}")
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """搜索知识库"""
        try:
            # 先检查 collection 是否存在
            if not self.milvus_client.has_collection(self.collection_name):
                print(f"Collection {self.collection_name} does not exist, returning empty results")
                return []
            
            query_embedding = self.processor.llm.embed(query)
            if not query_embedding:
                return []
            
            search_results = self.milvus_client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=limit,
                search_params={"metric_type": "IP", "params": {}},
                output_fields=["text", "source", "element_type", "metadata", "timestamp"]
            )
            
            results = []
            for res in search_results[0]:
                results.append({
                    "text": res["entity"]["text"],
                    "source": res["entity"]["source"],
                    "element_type": res["entity"]["element_type"],
                    "metadata": res["entity"]["metadata"] if res["entity"]["metadata"] else {},
                    "timestamp": res["entity"]["timestamp"],
                    "score": res["distance"]
                })
            
            return results
            
        except Exception as e:
            print(f"Error searching: {e}")
            return []
    
    def scan_directory(self, directory: str):
        """扫描目录并添加所有文档"""
        supported_extensions = {'.pdf', '.md', '.txt', '.docx', '.html'}
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if Path(file).suffix.lower() in supported_extensions:
                    file_path = os.path.join(root, file)
                    print(f"Processing {file_path}")
                    self.add_document(file_path)


class DocumentWatcher(FileSystemEventHandler):
    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base
        self.supported_extensions = {'.pdf', '.md', '.txt', '.docx', '.html'}
    
    def on_created(self, event):
        if not event.is_directory and Path(event.src_path).suffix.lower() in self.supported_extensions:
            print(f"New file detected: {event.src_path}")
            self.knowledge_base.add_document(event.src_path)
    
    def on_modified(self, event):
        if not event.is_directory and Path(event.src_path).suffix.lower() in self.supported_extensions:
            print(f"File modified: {event.src_path}")
            self.knowledge_base.add_document(event.src_path)
    
    def on_deleted(self, event):
        if not event.is_directory and Path(event.src_path).suffix.lower() in self.supported_extensions:
            print(f"File deleted: {event.src_path}")
            self.knowledge_base._remove_document(event.src_path)


def start_document_monitor(documents_dir: str = "documents"):
    """启动文档监控服务"""
    kb = KnowledgeBase()
    
    # 初始扫描
    if os.path.exists(documents_dir):
        print(f"Initial scan of {documents_dir}")
        kb.scan_directory(documents_dir)
    
    # 启动监控
    event_handler = DocumentWatcher(kb)
    observer = Observer()
    observer.schedule(event_handler, documents_dir, recursive=True)
    observer.start()
    
    print(f"Started monitoring {documents_dir}")
    
    try:
        while True:
            asyncio.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("Stopped document monitoring")
    
    observer.join()
    return kb


if __name__ == "__main__":
    kb = start_document_monitor()