"""
LlamaIndex RAG 服务

该模块提供了基于 LlamaIndex 的 RAG (Retrieval-Augmented Generation) 功能。
支持文档加载、索引创建、向量存储和查询。
"""

import logging
import os
import tempfile
import csv
import re
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# PDF处理相关导入
from pypdf import PdfReader

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
        # 创建 LLM 实例，使用 DeepSeek 模型
        from llama_index.llms.deepseek import DeepSeek
        import json
        
        # 使用 DeepSeek 专用的集成
        llm = DeepSeek(
            api_key=app_settings.DEEPSEEK_API_KEY,
            api_base=app_settings.DEEPSEEK_API_BASE,
            model=app_settings.DEEPSEEK_MODEL,  # 使用 DeepSeek 模型
            temperature=0.1
        )
        
        # 不使用原生客户端，回退到标准查询引擎
        self.use_deepseek_client = False
        logger.debug(f"\n\n==== 使用标准查询引擎 ====\n")
        
        logger.info(f"使用 DeepSeek 模型: {app_settings.DEEPSEEK_MODEL}")
        
        
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
        
        logger.debug(f"LlamaIndex 配置完成: model={app_settings.DEEPSEEK_MODEL}")
    
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
            
            # 记录文件大小
            file_size = len(file_content)
            logger.info(f"开始处理文件: {filename}, 大小: {file_size/1024/1024:.2f} MB")
            
            # 保存文件
            file_path = os.path.join(self.temp_dir, filename)
            with open(file_path, "wb") as f:
                f.write(file_content)
            
            # 对于大文件，特殊处理
            if file_size > 50 * 1024 * 1024:  # 大于50MB
                logger.warning(f"检测到大文件: {filename}, 将使用分块处理")
            
            # 检查文件扩展名
            if filename.lower().endswith('.csv'):
                # 使用特殊的CSV处理方法
                documents = await self._load_qa_csv(file_path)
                logger.info(f"使用CSV问答对加载器处理文件: {filename}")
            elif filename.lower().endswith('.pdf'):
                # 使用自定义PDF处理方法
                documents = await self._load_pdf_file(file_path)
                logger.info(f"使用自定义PDF加载器处理文件: {filename}")
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
            
            # 使用标准查询引擎获取相关文档
            response = query_engine.query(query_text)
            
            # 提取引用的源文档并添加引用编号
            source_nodes = response.source_nodes
            sources = []
            for i, node in enumerate(source_nodes, 1):
                sources.append({
                    "id": i,  # 添加引用ID
                    "text": node.text,
                    "score": node.score,
                    "metadata": node.metadata
                })
            
            # 默认使用标准查询引擎的响应
            answer = response.response
            thinking = None
            
            # 尝试使用 OpenAI 客户端直接调用 DeepSeek API 获取思维过程
            try:
                # 导入 OpenAI 客户端
                from openai import OpenAI
                
                # 构建上下文信息
                context = "\n\n".join([node["text"] for node in sources])
                
                # 创建 OpenAI 客户端连接 DeepSeek API
                client = OpenAI(
                    api_key=app_settings.DEEPSEEK_API_KEY,
                    base_url=app_settings.DEEPSEEK_API_BASE
                )
                
                # 构建针对兽医场景的英文提示
                prompt = f"""You are a professional veterinary assistant responding to questions from veterinarians. Please answer the question based on the information provided below. If the information is insufficient, supplement with your veterinary expertise.

When formulating your answer, consider how to help the veterinarian simplify the content for explaining to pet owners. Your response should be professional yet clear, and cite sources where appropriate.

For simple questions, provide direct and concise answers without unnecessarily diving too deep into the reference materials, even if additional information is available. Focus on what's most relevant to the specific question asked.

If the question is vague or lacks sufficient information for a proper diagnosis or recommendation, guide the veterinarian to provide more specific details. Clearly indicate what additional information would be helpful (such as pet's age, breed, weight, symptoms duration, previous medical history, etc.) to provide a more accurate and helpful response.

Citation format example: [1], [2], [3], etc., with complete references listed at the end of your response.

Information:
{context}

Question: {query_text}"""
                
                # 直接调用 DeepSeek API
                deepseek_response = client.chat.completions.create(
                    model=app_settings.DEEPSEEK_MODEL,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # 打印简洁的响应信息
                logger.debug(f"DeepSeek API 调用成功，模型: {app_settings.DEEPSEEK_MODEL}")
                
                # 获取思维过程
                if hasattr(deepseek_response.choices[0].message, 'reasoning_content'):
                    thinking = deepseek_response.choices[0].message.reasoning_content
                    logger.info(f"成功获取思维过程 (长度: {len(thinking)} 字符)")
                    # 使用 DeepSeek 的响应作为最终答案
                    answer = deepseek_response.choices[0].message.content
                else:
                    # 尝试从 __dict__ 中获取
                    message_dict = deepseek_response.choices[0].message.__dict__
                    if 'reasoning_content' in message_dict:
                        thinking = message_dict['reasoning_content']
                        logger.info(f"从消息字典获取到思维过程 (长度: {len(thinking)} 字符)")
                        # 使用 DeepSeek 的响应作为最终答案
                        answer = deepseek_response.choices[0].message.content
                    else:
                        logger.warning("未在响应中找到 reasoning_content 字段")
            except Exception as api_error:
                logger.error(f"直接调用 DeepSeek API 错误: {str(api_error)}")
                logger.info("回退到标准查询引擎的响应")
                
                # 尝试从标准响应中获取思维过程
                if hasattr(response, 'raw') and isinstance(response.raw, dict) and 'reasoning_content' in response.raw:
                    thinking = response.raw['reasoning_content']
                    logger.info(f"从原始响应获取到思维过程 (长度: {len(thinking)} 字符)")
                
                # 检查元数据
                elif hasattr(response, 'metadata') and response.metadata and 'reasoning_content' in response.metadata:
                    thinking = response.metadata['reasoning_content']
                    logger.info(f"从元数据获取到思维过程 (长度: {len(thinking)} 字符)")
            
            logger.info(f"查询集合 '{collection_name}': '{query_text[:30]}...'")
            
            # 构建响应结果
            result = {
                "answer": answer if 'answer' in locals() else response.response,
                "sources": sources
            }
            
            # 如果有思维过程，则添加到返回结果中
            if thinking:
                result["thinking"] = thinking
                logger.info(f"响应包含思维过程 (长度: {len(thinking)} 字符)")
            else:
                logger.warning(f"响应中未包含思维过程")
            
            return result
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
        
    async def delete_collection(self, collection_name: str) -> Dict[str, Any]:
        """
        删除一个集合
        
        参数:
            collection_name: 集合名称
            
        返回:
            删除结果
        """
        try:
            # 检查集合是否存在
            if collection_name in self.collections:
                # 从内存中移除
                del self.collections[collection_name]
            
            # 从ChromaDB中删除
            try:
                # 先检查集合是否存在
                collection_names = self.chroma_client.list_collections()
                if collection_name in collection_names:
                    self.chroma_client.delete_collection(collection_name)
                    logger.info(f"删除集合: {collection_name}")
                else:
                    logger.warning(f"集合 {collection_name} 不存在，无需删除")
            except Exception as e:
                logger.warning(f"从ChromaDB删除集合失败: {str(e)}")
            
            return {
                "status": "success",
                "message": f"集合 {collection_name} 已删除"
            }
        except Exception as e:
            logger.error(f"删除集合失败: {str(e)}")
            raise
            
    async def list_collections(self) -> List[Dict[str, Any]]:
        """
        获取所有集合的列表
        
        返回:
            集合列表，包含每个集合的名称和文档数量
        """
        try:
            # 获取所有集合
            collections_info = []
            # ChromaDB v0.6.0只返回集合名称列表
            collection_names = self.chroma_client.list_collections()
            
            for collection_name in collection_names:
                try:
                    # 在v0.6.0中，collection_name直接是字符串
                    name = collection_name
                    
                    # 获取每个集合的信息
                    chroma_collection = self.chroma_client.get_collection(name)
                    
                    # 获取文档数量
                    count = chroma_collection.count()
                    
                    # 获取文件信息汇总
                    file_summary = {}
                    if count > 0:
                        try:
                            # 查询文档元数据
                            results = chroma_collection.get(limit=count)
                            if results and 'metadatas' in results and results['metadatas']:
                                for metadata in results['metadatas']:
                                    if metadata and isinstance(metadata, dict):
                                        # 提取文件名
                                        file_name = metadata.get("file_name", "Unknown")
                                        file_type = metadata.get("file_type", "Unknown")
                                        
                                        # 汇总文件信息
                                        if file_name not in file_summary:
                                            file_summary[file_name] = {
                                                "file_name": file_name,
                                                "file_type": file_type,
                                                "file_size": metadata.get("file_size", 0),
                                                "count": 1
                                            }
                                        else:
                                            file_summary[file_name]["count"] += 1
                        except Exception as e:
                            logger.warning(f"获取集合 {name} 的元数据汇总失败: {str(e)}")
                    
                    # 添加集合信息
                    collections_info.append({
                        "name": name,
                        "document_count": count,
                        "files": list(file_summary.values())
                    })
                except Exception as e:
                    logger.warning(f"获取集合 {name} 信息失败: {str(e)}")
                    collections_info.append({
                        "name": name,
                        "document_count": 0,
                        "error": str(e)
                    })
            
            logger.info(f"获取到 {len(collections_info)} 个集合")
            return collections_info
        except Exception as e:
            logger.error(f"获取集合列表失败: {str(e)}")
            raise
            
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        获取指定集合的详细信息
        
        参数:
            collection_name: 集合名称
            
        返回:
            集合信息，包含文档数量和文件汇总
        """
        try:
            # 检查集合是否存在
            try:
                # 尝试获取集合
                chroma_collection = self.chroma_client.get_collection(collection_name)
            except Exception as e:
                logger.warning(f"集合 {collection_name} 不存在: {str(e)}")
                return {
                    "name": collection_name,
                    "exists": False,
                    "document_count": 0,
                    "files": []
                }
            
            # 获取文档数量
            count = chroma_collection.count()
            
            # 获取文件汇总
            file_summary = {}
            if count > 0:
                try:
                    # 获取所有文档的元数据
                    results = chroma_collection.get(limit=count)
                    if results and 'metadatas' in results and results['metadatas']:
                        for metadata in results['metadatas']:
                            if metadata and isinstance(metadata, dict):
                                # 提取文件名
                                file_name = metadata.get("file_name", "Unknown")
                                file_type = metadata.get("file_type", "Unknown")
                                
                                # 汇总文件信息
                                if file_name not in file_summary:
                                    file_summary[file_name] = {
                                        "file_name": file_name,
                                        "file_type": file_type,
                                        "file_size": metadata.get("file_size", 0),
                                        "chunk_count": 1
                                    }
                                else:
                                    file_summary[file_name]["chunk_count"] += 1
                except Exception as e:
                    logger.warning(f"获取集合 {collection_name} 的文件汇总失败: {str(e)}")
            
            logger.info(f"获取集合 {collection_name} 信息，包含 {count} 个文档和 {len(file_summary)} 个文件")
            
            # 格式化文件大小
            for file_info in file_summary.values():
                if "file_size" in file_info and file_info["file_size"] > 0:
                    size_mb = file_info["file_size"] / (1024 * 1024)
                    file_info["file_size_formatted"] = f"{size_mb:.2f} MB"
            
            return {
                "name": collection_name,
                "exists": True,
                "document_count": count,
                "file_count": len(file_summary),
                "files": list(file_summary.values())
            }
        except Exception as e:
            logger.error(f"获取集合信息失败: {str(e)}")
            raise
    
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
    
    async def _load_pdf_file(self, file_path: str) -> List[Document]:
        """
        加载PDF文件，使用自定义处理方法提取文本并分割成文档
        
        参数:
            file_path: PDF文件路径
            
        返回:
            文档列表
        """
        documents = []
        try:
            # 使用pypdf读取PDF文件
            reader = PdfReader(file_path)
            
            # 提取所有页面的文本
            all_text = ""
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    # 清理文本，移除PDF结构标记和不必要的字符
                    text = self._clean_pdf_text(text)
                    all_text += text + "\n\n"
            
            # 将整个文本作为一个文档，让LlamaIndex的分词器处理分割
            if all_text.strip():
                metadata = {
                    "source": file_path,
                    "file_path": file_path,
                    "file_name": os.path.basename(file_path),
                    "file_type": "application/pdf",
                    "file_size": os.path.getsize(file_path),
                    "creation_date": "2025-05-06",  # 使用当前日期
                    "last_modified_date": "2025-05-06",  # 使用当前日期
                }
                
                # 创建文档对象
                doc = Document(text=all_text, metadata=metadata)
                documents.append(doc)
                
                logger.info(f"从PDF文件提取了{len(all_text)}个字符的文本")
            else:
                logger.warning(f"PDF文件未提取到任何文本内容")
                
            return documents
        except Exception as e:
            logger.error(f"加载PDF文件失败: {str(e)}")
            raise
    
    def _clean_pdf_text(self, text: str) -> str:
        """
        清理从PDF提取的文本，移除结构标记和不必要的字符
        
        参数:
            text: 从PDF提取的原始文本
            
        返回:
            清理后的文本
        """
        # 移除PDF结构标记，如对象引用、链接定义等
        text = re.sub(r'\rendobj\r\d+ \d+ obj\r', ' ', text)
        text = re.sub(r'<</[^>]+>>', ' ', text)
        text = re.sub(r'/Type/[\w]+', ' ', text)
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除其他可能的PDF结构标记
        text = re.sub(r'\[\d+ \d+ R\]', ' ', text)
        
        return text.strip()
    
    def __del__(self):
        """析构函数"""
        self.cleanup()
