"""
LlamaIndex RAG 服务

该模块提供了基于 LlamaIndex 的 RAG (Retrieval-Augmented Generation) 功能。
支持文档加载、索引创建、向量存储和查询。
"""

import logging
import os
import tempfile
import csv
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from app.core.config import settings as app_settings

logger = logging.getLogger(__name__)

class LlamaIndexRAGService:
    """
    LlamaIndex RAG 服务类
    
    提供基于 LlamaIndex 的 RAG 功能，包括：
    - 文档加载和处理
    - 索引创建和管理
    - 向量存储
    - 查询和检索
    """
    
    def __init__(self):
        """初始化 LlamaIndex RAG 服务"""
        # 设置更详细的日志
        logger.setLevel(logging.DEBUG)
        logger.debug("初始化 LlamaIndex RAG 服务")
        
        # 设置 LlamaIndex 全局配置
        self._setup_llama_index()
        
        # 初始化向量存储
        self._setup_vector_store()
        
        # 存储临时文件路径
        self.temp_dir = None
    
    def _setup_llama_index(self):
        """设置 LlamaIndex 全局配置"""
        # 创建 LLM 实例，使用 OpenRouter 提供的 OpenAI 模型
        llm = OpenAI(
            api_key=app_settings.OPENAI_API_KEY,
            api_base=app_settings.OPENAI_API_BASE,
            model=app_settings.LLAMA_INDEX_MODEL,  # 使用 OpenRouter 支持的 OpenAI 模型
            temperature=0.1,
            extra_headers={
                "HTTP-Referer": "https://yanhuobu.com",  # 替换为实际域名
                "X-Title": "烟火簿"
            }
        )
        
        # 设置环境变量，强制使用本地模型
        import os
        os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 强制离线模式
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_CACHE"] = "./huggingface"
        os.environ["HF_HOME"] = "./huggingface"
        
        # 创建嵌入模型实例 - 使用 HuggingFace 的模型
        try:
            # 查找实际的快照目录
            model_path = "./huggingface/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/"
            import glob
            snapshot_dirs = glob.glob(f"{model_path}*")
            if snapshot_dirs:
                actual_model_path = snapshot_dirs[0]
                logger.debug(f"使用本地模型路径: {actual_model_path}")
                # 直接使用本地模型路径
                embed_model = HuggingFaceEmbedding(
                    model_name=actual_model_path
                )
            else:
                logger.warning("未找到本地模型快照，将尝试使用缓存目录")
                embed_model = HuggingFaceEmbedding(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    cache_folder="./huggingface"
                )
        except Exception as e:
            logger.error(f"加载本地模型失败: {str(e)}")
            # 回退到在线模式
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            os.environ.pop("HF_DATASETS_OFFLINE", None)
            embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                cache_folder="./huggingface"
            )
        
        # 创建文本分块器
        node_parser = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=200
        )
        
        # 设置 LlamaIndex 全局配置
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.node_parser = node_parser
        
        logger.debug(f"LlamaIndex 配置完成: model={app_settings.LLAMA_INDEX_MODEL}")
    
    def _setup_vector_store(self):
        """设置向量存储"""
        # 使用 ChromaDB 作为向量存储
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collections = {}
    
    async def create_collection(self, collection_name: str) -> Any:
        """
        创建或获取一个集合
        
        参数:
            collection_name: 集合名称
            
        返回:
            集合对象
        """
        try:
            # 检查集合是否已存在
            if collection_name in self.collections:
                return self.collections[collection_name]
            
            # 创建或获取集合
            chroma_collection = self.chroma_client.get_or_create_collection(collection_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            
            # 创建索引
            index = VectorStoreIndex.from_vector_store(vector_store)
            
            # 存储索引
            self.collections[collection_name] = index
            
            logger.info(f"创建集合: {collection_name}")
            return index
        except Exception as e:
            logger.error(f"创建集合失败: {str(e)}")
            raise
    
    async def add_documents(self, collection_name: str, documents: List[Document]) -> Any:
        """
        向集合添加文档
        
        参数:
            collection_name: 集合名称
            documents: 文档列表
            
        返回:
            更新后的索引
        """
        try:
            # 获取或创建集合
            index = await self.create_collection(collection_name)
            
            # 添加文档
            for doc in documents:
                index.insert(doc)
            
            logger.info(f"向集合 {collection_name} 添加了 {len(documents)} 个文档")
            return index
        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            raise
    
    async def upload_file(self, file_content: bytes, filename: str, collection_name: str) -> Dict[str, Any]:
        """
        上传文件并添加到集合
        
        参数:
            file_content: 文件内容（字节）
            filename: 文件名
            collection_name: 集合名称
            
        返回:
            上传结果
        """
        try:
            # 创建临时目录
            if self.temp_dir is None:
                self.temp_dir = tempfile.mkdtemp()
            
            # 保存文件
            file_path = os.path.join(self.temp_dir, filename)
            with open(file_path, "wb") as f:
                f.write(file_content)
            
            # 检查文件扩展名
            if filename.lower().endswith('.csv'):
                # 使用特殊的CSV处理方法
                documents = await self._load_qa_csv(file_path)
                logger.info(f"使用CSV问答对加载器处理文件: {filename}")
            else:
                # 使用标准文档加载器
                documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
            
            # 添加到集合
            await self.add_documents(collection_name, documents)
            
            logger.info(f"上传文件 {filename} 到集合 {collection_name}")
            return {
                "filename": filename,
                "collection": collection_name,
                "document_count": len(documents)
            }
        except Exception as e:
            logger.error(f"上传文件失败: {str(e)}")
            raise
    
    async def query(self, collection_name: str, query_text: str, similarity_top_k: int = 3) -> Dict[str, Any]:
        """
        查询集合
        
        参数:
            collection_name: 集合名称
            query_text: 查询文本
            similarity_top_k: 返回的相似文档数量
            
        返回:
            查询结果
        """
        try:
            # 获取集合
            if collection_name not in self.collections:
                await self.create_collection(collection_name)
            
            index = self.collections[collection_name]
            
            # 创建查询引擎
            query_engine = index.as_query_engine(similarity_top_k=similarity_top_k)
            
            # 执行查询
            response = query_engine.query(query_text)
            
            # 提取引用的源文档
            source_nodes = response.source_nodes
            sources = []
            for node in source_nodes:
                sources.append({
                    "text": node.text,
                    "score": node.score,
                    "metadata": node.metadata
                })
            
            logger.info(f"查询集合 {collection_name}: {query_text}")
            return {
                "answer": response.response,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"查询失败: {str(e)}")
            raise
    
    def cleanup(self):
        """清理资源"""
        # 清理临时目录
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
        
        logger.debug("清理资源完成")
    
    async def _load_qa_csv(self, file_path: str) -> List[Document]:
        """
        加载CSV问答对文件，每一行作为一个独立的文档
        
        参数:
            file_path: CSV文件路径
            
        返回:
            文档列表
        """
        documents = []
        try:
            # 检测文件编码
            encoding = 'utf-8-sig' if os.path.getsize(file_path) >= 3 and open(file_path, 'rb').read(3) == b'\xef\xbb\xbf' else 'utf-8'
            
            with open(file_path, 'r', encoding=encoding, newline='') as csvfile:
                reader = csv.reader(csvfile)
                for i, row in enumerate(reader):
                    if len(row) >= 2:  # 确保至少有问题和答案两列
                        question = row[0].strip()
                        answer = row[1].strip()
                        
                        # 创建文档，将问题和答案作为一个整体
                        content = f"问题: {question}\n答案: {answer}"
                        metadata = {
                            "source": file_path,
                            "row": i + 1,
                            "question": question,
                            "answer": answer,
                            "type": "qa_pair"
                        }
                        
                        # 创建文档对象，每行作为一个独立文档
                        doc = Document(text=content, metadata=metadata)
                        documents.append(doc)
                    else:
                        logger.warning(f"CSV文件第{i+1}行格式不正确，跳过")
            
            logger.info(f"从CSV文件加载了 {len(documents)} 个问答对")
            return documents
        except Exception as e:
            logger.error(f"加载CSV问答对失败: {str(e)}")
            raise
    
    def __del__(self):
        """析构函数"""
        self.cleanup()
