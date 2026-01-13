"""
函数式编程版本的知识库模块
使用纯函数、函数组合和不可变数据结构
"""

import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from functools import partial, reduce

from pymilvus import MilvusClient, DataType
from dotenv import load_dotenv
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.md import partition_md

# 导入函数式 LLM 模块
import sys
sys.path.append(str(Path(__file__).parent))
from local_llm_direct import (
    create_config, embed_text, LLMConfig, EmbeddingResult
)

load_dotenv()


# ============================================================================
# 数据类型定义 - 使用不可变的 dataclass
# ============================================================================

@dataclass(frozen=True)
class DocumentChunk:
    """不可变的文档块"""
    id: str
    text: str
    source: str
    element_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddedChunk:
    """不可变的嵌入文档块"""
    chunk: DocumentChunk
    embedding: List[float]


@dataclass(frozen=True)
class SearchResult:
    """不可变的搜索结果"""
    text: str
    source: str
    element_type: str
    metadata: Dict[str, Any]
    timestamp: str
    score: float


@dataclass(frozen=True)
class FileHash:
    """不可变的文件哈希记录"""
    file_path: str
    hash_value: str


@dataclass(frozen=True)
class CollectionConfig:
    """不可变的集合配置"""
    milvus_uri: str
    collection_name: str
    index_type: str = "HNSW"
    metric_type: str = "COSINE"
    m: int = 64
    ef_construction: int = 100


@dataclass(frozen=True)
class ProcessingConfig:
    """不可变的文档处理配置"""
    supported_extensions: Tuple[str, ...] = ('.pdf', '.md', '.txt', '.docx', '.html')
    pdf_languages: Tuple[str, ...] = ("chi_sim", "eng")
    pdf_strategy: str = "hi_res"
    md_max_characters: int = 2000
    md_new_after_n_chars: int = 1500


# ============================================================================
# 纯函数 - 文件操作相关
# ============================================================================

def compute_file_hash(file_path: str) -> str:
    """计算文件哈希 - 副作用（读取文件）"""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def load_file_hashes(hash_file: str = "file_hashes.json") -> Dict[str, str]:
    """加载文件哈希记录 - 副作用（读取文件）"""
    if os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            return json.load(f)
    return {}


def save_file_hashes(hashes: Dict[str, str], hash_file: str = "file_hashes.json") -> None:
    """保存文件哈希记录 - 副作用（写入文件）"""
    with open(hash_file, 'w') as f:
        json.dump(hashes, f, indent=2)


def file_has_changed(file_path: str, stored_hashes: Dict[str, str]) -> bool:
    """检查文件是否变更 - 纯函数"""
    current_hash = compute_file_hash(file_path)
    stored_hash = stored_hashes.get(file_path)
    return current_hash != stored_hash


def update_file_hash(file_path: str, hashes: Dict[str, str]) -> Dict[str, str]:
    """更新文件哈希 - 返回新的哈希字典（不可变）"""
    new_hash = compute_file_hash(file_path)
    return {**hashes, file_path: new_hash}


def is_supported_file(file_path: str, config: ProcessingConfig) -> bool:
    """检查文件是否支持 - 纯函数"""
    return Path(file_path).suffix.lower() in config.supported_extensions


def collect_files_in_directory(
    directory: str,
    config: ProcessingConfig
) -> List[str]:
    """收集目录中的所有支持文件 - 副作用（读取目录）"""
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            if is_supported_file(file_path, config):
                files.append(file_path)
    return files


# ============================================================================
# 纯函数 - 数据转换
# ============================================================================

def sanitize_metadata(metadata: Any) -> Dict[str, Any]:
    """清洗 metadata - 纯函数"""
    if not isinstance(metadata, dict):
        return {}

    sanitized = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[key] = value
        else:
            sanitized[key] = str(value)

    return sanitized


def element_to_chunk(element: Any, file_path: str, index: int) -> Optional[DocumentChunk]:
    """将 element 转换为文档块 - 纯函数"""
    if not (hasattr(element, 'text') and element.text.strip()):
        return None

    metadata = {}
    if hasattr(element, 'metadata'):
        metadata = element.metadata.to_dict()

    return DocumentChunk(
        id=f"{Path(file_path).stem}_{index}",
        text=element.text,
        source=file_path,
        element_type=str(type(element).__name__),
        metadata=sanitize_metadata(metadata)
    )


def elements_to_chunks(elements: List[Any], file_path: str) -> List[DocumentChunk]:
    """将 elements 列表转换为文档块列表 - 纯函数"""
    chunks = []
    for i, element in enumerate(elements):
        chunk = element_to_chunk(element, file_path, i)
        if chunk:
            chunks.append(chunk)
    return chunks


def chunk_to_milvus_data(
    chunk: DocumentChunk,
    embedding: List[float]
) -> Dict[str, Any]:
    """将文档块转换为 Milvus 数据格式 - 纯函数"""
    return {
        "vector": embedding,
        "text": chunk.text,
        "source": chunk.source,
        "element_type": chunk.element_type,
        "metadata": chunk.metadata,
        "timestamp": datetime.now().isoformat(),
        "chunk_id": chunk.id
    }


def search_result_to_dict(result: Dict[str, Any]) -> SearchResult:
    """将 Milvus 搜索结果转换为 SearchResult - 纯函数"""
    entity = result["entity"]
    return SearchResult(
        text=entity["text"],
        source=entity["source"],
        element_type=entity["element_type"],
        metadata=entity.get("metadata", {}),
        timestamp=entity["timestamp"],
        score=result["distance"]
    )


# ============================================================================
# 副作用函数 - 文档处理
# ============================================================================

def process_pdf_document(file_path: str, config: ProcessingConfig) -> List[Any]:
    """处理 PDF 文档 - 副作用（读取文件）"""
    return partition_pdf(
        filename=file_path,
        languages=list(config.pdf_languages),
        strategy=config.pdf_strategy,
        extract_images=False,
        infer_table_structure=True,
        ocr_mode="entire_page",
        extract_image_block_to_payload=False,
    )


def process_md_document(file_path: str, config: ProcessingConfig) -> List[Any]:
    """处理 Markdown 文档 - 副作用（读取文件）"""
    return partition_md(
        filename=file_path,
        encoding="utf-8",
        languages=list(config.pdf_languages),
        chunking_strategy="by_title",
        max_characters=config.md_max_characters,
        new_after_n_chars=config.md_new_after_n_chars,
    )


def process_generic_document(file_path: str) -> List[Any]:
    """处理通用文档 - 副作用（读取文件）"""
    return partition(filename=file_path)


def parse_document(file_path: str, config: ProcessingConfig) -> List[Any]:
    """解析文档 - 副作用（读取文件）"""
    try:
        if file_path.lower().endswith(".pdf"):
            print("使用partition_pdf")
            return process_pdf_document(file_path, config)
        elif file_path.lower().endswith(".md"):
            print("使用partition_md")
            return process_md_document(file_path, config)
        else:
            print("使用通用的partition")
            return process_generic_document(file_path)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []


def process_document(file_path: str, config: ProcessingConfig) -> List[DocumentChunk]:
    """
    处理文档并返回文档块列表
    副作用函数：读取文件
    """
    elements = parse_document(file_path, config)
    return elements_to_chunks(elements, file_path)


# ============================================================================
# 副作用函数 - 嵌入生成
# ============================================================================

def embed_chunk(
    chunk: DocumentChunk,
    llm_config: LLMConfig
) -> Optional[EmbeddedChunk]:
    """
    为文档块生成嵌入
    副作用函数：调用 LLM API
    """
    result = embed_text(llm_config, chunk.text)

    if result.is_success and result.embedding:
        return EmbeddedChunk(chunk=chunk, embedding=result.embedding)
    else:
        print(f"Failed to embed chunk {chunk.id}: {result.error}")
        return None


def embed_chunks(
    chunks: List[DocumentChunk],
    llm_config: LLMConfig
) -> List[EmbeddedChunk]:
    """
    为文档块列表生成嵌入
    副作用函数：调用 LLM API
    """
    embedded = []
    for chunk in chunks:
        embedded_chunk = embed_chunk(chunk, llm_config)
        if embedded_chunk:
            embedded.append(embedded_chunk)
    return embedded


# ============================================================================
# 副作用函数 - Milvus 数据库操作
# ============================================================================

def create_test_embedding(llm_config: LLMConfig) -> List[float]:
    """创建测试嵌入以获取维度 - 副作用（调用 API）"""
    result = embed_text(llm_config, "test")
    if result.is_success:
        return result.embedding
    else:
        raise RuntimeError(f"Failed to create test embedding: {result.error}")


def initialize_collection(
    client: MilvusClient,
    config: CollectionConfig,
    llm_config: LLMConfig
) -> None:
    """
    初始化 Milvus 集合
    副作用函数：创建数据库集合
    """
    if client.has_collection(config.collection_name):
        print(f"Collection {config.collection_name} already exists")
        return

    # 获取嵌入维度
    test_embedding = create_test_embedding(llm_config)
    dimension = len(test_embedding)

    # 定义 Schema
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=512)

    # 定义 Index
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type=config.index_type,
        index_name="vector_index",
        metric_type=config.metric_type,
        params={
            "M": config.m,
            "efConstruction": config.ef_construction
        }
    )

    # 创建集合
    client.create_collection(
        collection_name=config.collection_name,
        schema=schema,
        index_params=index_params,
        consistency_level="Bounded"
    )

    # 加载集合到内存
    client.load_collection(collection_name=config.collection_name)
    print(f"创建collection: {config.collection_name}")


def insert_embedded_chunks(
    client: MilvusClient,
    collection_name: str,
    embedded_chunks: List[EmbeddedChunk]
) -> int:
    """
    插入嵌入文档块到 Milvus
    副作用函数：写入数据库
    返回插入的数量
    """
    if not embedded_chunks:
        return 0

    data = [
        chunk_to_milvus_data(ec.chunk, ec.embedding)
        for ec in embedded_chunks
    ]

    client.insert(collection_name=collection_name, data=data)
    return len(data)


def remove_document_from_collection(
    client: MilvusClient,
    collection_name: str,
    file_path: str
) -> int:
    """
    从集合中删除指定文档的所有块
    副作用函数：删除数据库记录
    返回删除的数量
    """
    try:
        if not client.has_collection(collection_name):
            print(f"Collection {collection_name} does not exist")
            return 0

        # 查找该文件的所有记录
        search_results = client.query(
            collection_name=collection_name,
            filter=f'source == "{file_path}"',
            output_fields=["id"]
        )

        if search_results:
            ids_to_delete = [result["id"] for result in search_results]
            client.delete(
                collection_name=collection_name,
                ids=ids_to_delete
            )
            print(f"Removed {len(ids_to_delete)} chunks from {file_path}")
            return len(ids_to_delete)

        return 0

    except Exception as e:
        print(f"Error removing document {file_path}: {e}")
        return 0


def search_collection(
    client: MilvusClient,
    collection_name: str,
    query_embedding: List[float],
    limit: int = 5
) -> List[SearchResult]:
    """
    搜索集合
    副作用函数：查询数据库
    """
    try:
        if not client.has_collection(collection_name):
            print(f"Collection {collection_name} does not exist")
            return []

        search_results = client.search(
            collection_name=collection_name,
            anns_field="vector",
            data=[query_embedding],
            limit=limit,
            search_params={"params": {"ef": 10}},
            output_fields=["text", "source", "element_type", "metadata", "timestamp"]
        )

        return [search_result_to_dict(res) for res in search_results[0]]

    except Exception as e:
        print(f"Error searching: {e}")
        return []


def drop_collection(client: MilvusClient, collection_name: str) -> bool:
    """
    删除集合
    副作用函数：删除数据库集合
    """
    try:
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)
            print(f"Collection {collection_name} has been dropped")
            return True
        else:
            print(f"Collection {collection_name} does not exist")
            return False
    except Exception as e:
        print(f"Error dropping collection {collection_name}: {e}")
        return False


# ============================================================================
# 高级函数 - 组合多个操作
# ============================================================================

def add_document_to_kb(
    client: MilvusClient,
    collection_name: str,
    file_path: str,
    llm_config: LLMConfig,
    processing_config: ProcessingConfig,
    file_hashes: Dict[str, str]
) -> Tuple[bool, Dict[str, str]]:
    """
    添加文档到知识库
    副作用函数：组合多个操作
    返回 (是否成功, 更新后的哈希字典)
    """
    try:
        # 检查文件是否变更
        if not file_has_changed(file_path, file_hashes):
            print(f"File {file_path} unchanged, skipping")
            return (False, file_hashes)

        # 处理文档
        chunks = process_document(file_path, processing_config)
        if not chunks:
            print(f"No chunks extracted from {file_path}")
            return (False, file_hashes)

        # 删除旧记录
        remove_document_from_collection(client, collection_name, file_path)

        # 生成嵌入
        embedded_chunks = embed_chunks(chunks, llm_config)
        if not embedded_chunks:
            print(f"No embeddings generated for {file_path}")
            return (False, file_hashes)

        # 插入数据
        count = insert_embedded_chunks(client, collection_name, embedded_chunks)

        # 更新哈希
        new_hashes = update_file_hash(file_path, file_hashes)

        print(f"Added {count} chunks from {file_path}")
        return (True, new_hashes)

    except Exception as e:
        print(f"Error adding document {file_path}: {e}")
        return (False, file_hashes)


def search_kb(
    client: MilvusClient,
    collection_name: str,
    query: str,
    llm_config: LLMConfig,
    limit: int = 5
) -> List[SearchResult]:
    """
    搜索知识库
    副作用函数：组合查询操作
    """
    # 生成查询嵌入
    result = embed_text(llm_config, query)

    if not result.is_success or not result.embedding:
        print(f"Failed to embed query: {result.error}")
        return []

    # 搜索
    return search_collection(client, collection_name, result.embedding, limit)


def scan_and_add_directory(
    client: MilvusClient,
    collection_name: str,
    directory: str,
    llm_config: LLMConfig,
    processing_config: ProcessingConfig,
    file_hashes: Dict[str, str]
) -> Dict[str, str]:
    """
    扫描目录并添加所有文档
    副作用函数：批量处理文档
    返回更新后的哈希字典
    """
    files = collect_files_in_directory(directory, processing_config)

    current_hashes = file_hashes
    for file_path in files:
        print(f"Processing {file_path}")
        success, new_hashes = add_document_to_kb(
            client,
            collection_name,
            file_path,
            llm_config,
            processing_config,
            current_hashes
        )
        current_hashes = new_hashes

    return current_hashes


# ============================================================================
# 便捷函数 - 创建预配置的函数
# ============================================================================

def create_kb_context(
    milvus_uri: str = "./milvus_demo.db",
    collection_name: str = "finance_knowledge",
    model: str = 'qwen3:0.6b',
    embed_model: str = 'qwen3-embedding:0.6b'
) -> Tuple[MilvusClient, CollectionConfig, LLMConfig, ProcessingConfig]:
    """
    创建知识库上下文
    返回所有必要的配置和客户端
    """
    # 创建客户端
    client = MilvusClient(uri=milvus_uri)

    # 创建配置
    collection_config = CollectionConfig(
        milvus_uri=milvus_uri,
        collection_name=collection_name
    )

    llm_config = create_config(model=model, embed_model=embed_model)

    processing_config = ProcessingConfig()

    # 初始化集合
    initialize_collection(client, collection_config, llm_config)

    return (client, collection_config, llm_config, processing_config)


def create_add_document_function(
    client: MilvusClient,
    collection_name: str,
    llm_config: LLMConfig,
    processing_config: ProcessingConfig
) -> Callable:
    """创建预配置的添加文档函数"""
    return partial(
        add_document_to_kb,
        client=client,
        collection_name=collection_name,
        llm_config=llm_config,
        processing_config=processing_config
    )


def create_search_function(
    client: MilvusClient,
    collection_name: str,
    llm_config: LLMConfig
) -> Callable:
    """创建预配置的搜索函数"""
    return partial(
        search_kb,
        client=client,
        collection_name=collection_name,
        llm_config=llm_config
    )


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 创建知识库上下文
    client, coll_config, llm_config, proc_config = create_kb_context(
        collection_name="finance_knowledge_functional"
    )

    # 加载文件哈希
    file_hashes = load_file_hashes()

    # 扫描并添加文档
    documents_dir = "documents"
    if os.path.exists(documents_dir):
        print(f"扫描目录: {documents_dir}")
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

    # 测试搜索
    print("\n测试搜索:")
    queries = [
        "今年国庆节消费怎么样？",
        "4月关税对中国股市的冲击?"
    ]

    for query in queries:
        print(f"\n查询: {query}")
        results = search_kb(
            client,
            coll_config.collection_name,
            query,
            llm_config,
            limit=2
        )

        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.text[:100]}... (相关度: {result.score:.3f})")
