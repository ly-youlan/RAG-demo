"""
LlamaIndex RAG API 路由

该模块提供了基于 LlamaIndex 的 RAG 功能的 API 路由。
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks, Request
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import logging

from app.ai.llama_index_rag import LlamaIndexRAGService

# 创建路由器
router = APIRouter(prefix="/api/v1/rag/llama-index", tags=["RAG-LlamaIndex"])
logger = logging.getLogger(__name__)

# 依赖注入
async def get_llama_index_service():
    """
    获取 LlamaIndex RAG 服务实例
    
    返回:
        LlamaIndexRAGService 实例
    """
    service = LlamaIndexRAGService()
    try:
        yield service
    finally:
        service.cleanup()

# 请求和响应模型
class QueryRequest(BaseModel):
    """查询请求模型"""
    collection_name: str
    query: str
    top_k: Optional[int] = 3

class CollectionResponse(BaseModel):
    """集合响应模型"""
    name: str
    document_count: int

class Source(BaseModel):
    """引用源模型"""
    id: int
    text: str
    score: float
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    """查询响应模型"""
    answer: str
    sources: List[Source]
    thinking: Optional[str] = None

@router.get("/collections", tags=["RAG-LlamaIndex"])
async def list_collections(
    llama_index_service: LlamaIndexRAGService = Depends(get_llama_index_service)
):
    """
    获取所有集合的列表
    
    返回所有集合的名称和文档数量
    """
    try:
        # 获取集合列表
        result = await llama_index_service.list_collections()
        return result
    except Exception as e:
        logger.error(f"获取集合列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取集合列表失败: {str(e)}")

@router.get("/collections/{collection_name}", tags=["RAG-LlamaIndex"])
async def get_collection_info(
    collection_name: str,
    llama_index_service: LlamaIndexRAGService = Depends(get_llama_index_service)
):
    """
    获取指定集合的详细信息
    
    参数:
    - collection_name: 集合名称
    """
    try:
        # 获取集合信息
        result = await llama_index_service.get_collection_info(collection_name)
        return result
    except Exception as e:
        logger.error(f"获取集合信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取集合信息失败: {str(e)}")

@router.delete("/collections/{collection_name}", tags=["RAG-LlamaIndex"])
async def delete_collection(
    collection_name: str,
    llama_index_service: LlamaIndexRAGService = Depends(get_llama_index_service)
):
    """
    删除指定集合
    
    参数:
    - collection_name: 集合名称
    """
    try:
        # 删除集合
        result = await llama_index_service.delete_collection(collection_name)
        return result
    except Exception as e:
        logger.error(f"删除集合失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除集合失败: {str(e)}")


@router.post("/collections/{collection_name}/upload", tags=["RAG-LlamaIndex"])
async def upload_file(
    collection_name: str,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    llama_index_service: LlamaIndexRAGService = Depends(get_llama_index_service)
):
    """
    上传文件到指定集合
    
    参数:
    - collection_name: 集合名称
    - file: 上传的文件
    """
    try:
        # 记录开始上传时间
        import time
        start_time = time.time()
        logger.info(f"开始上传文件: {file.filename} 到集合: {collection_name}")
        
        # 分块读取文件内容，避免一次加载大文件到内存
        # 对于大文件，我们先写入临时文件，然后处理
        import tempfile
        import os
        import shutil
        
        # 创建临时文件
        temp_file_path = None
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                temp_file_path = temp_file.name
                
                # 分块读取和写入
                chunk_size = 1024 * 1024  # 1MB chunks
                total_size = 0
                while True:
                    chunk = await file.read(chunk_size)
                    if not chunk:
                        break
                    temp_file.write(chunk)
                    total_size += len(chunk)
                    
                    # 每读取10MB记录一次进度
                    if total_size % (10 * 1024 * 1024) == 0:  # 每10MB记录一次
                        logger.info(f"文件上传进度: {total_size/1024/1024:.2f} MB")
            
            # 记录文件大小
            logger.info(f"文件上传完成: {total_size/1024/1024:.2f} MB, 耗时: {time.time() - start_time:.2f} 秒")
            
            # 读取文件内容
            with open(temp_file_path, "rb") as f:
                file_content = f.read()
            
            # 上传文件到集合
            result = await llama_index_service.upload_file(
                file_content=file_content,
                filename=file.filename,
                collection_name=collection_name
            )
            
            # 添加后台任务清理资源
            if background_tasks:
                async def cleanup_resources():
                    try:
                        llama_index_service.cleanup()
                        # 清理临时文件
                        if temp_file_path and os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                    except Exception as e:
                        logger.error(f"清理资源失败: {str(e)}")
                
                background_tasks.add_task(cleanup_resources)
            
            # 记录完成时间
            logger.info(f"文件处理完成: {file.filename}, 总耗时: {time.time() - start_time:.2f} 秒")
            return result
        finally:
            # 确保临时文件被清理
            if temp_file_path and os.path.exists(temp_file_path) and not background_tasks:
                try:
                    os.remove(temp_file_path)
                except Exception as e:
                    logger.error(f"删除临时文件失败: {str(e)}")
    except Exception as e:
        logger.error(f"上传文件失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"上传文件失败: {str(e)}")


@router.post("/query", response_model=QueryResponse, tags=["RAG-LlamaIndex"])
async def query(
    request: QueryRequest,
    background_tasks: BackgroundTasks = None,
    llama_index_service: LlamaIndexRAGService = Depends(get_llama_index_service)
):
    """
    查询集合
    
    参数:
    - collection_name: 集合名称
    - query: 查询文本
    - top_k: 返回的相似文档数量
    """
    try:
        # 执行查询
        result = await llama_index_service.query(
            collection_name=request.collection_name,
            query_text=request.query,
            similarity_top_k=request.top_k
        )
        
        # 添加后台任务清理资源
        if background_tasks:
            async def cleanup_resources():
                try:
                    llama_index_service.cleanup()
                except Exception as e:
                    logger.error(f"清理资源失败: {str(e)}")
            
            background_tasks.add_task(cleanup_resources)
        
        return result
    except Exception as e:
        logger.error(f"查询失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

@router.post("/query-with-file", tags=["RAG-LlamaIndex"])
async def query_with_file(
    request: Request,
    background_tasks: BackgroundTasks,
    llama_index_service: LlamaIndexRAGService = Depends(get_llama_index_service)
):
    """
    上传文件并立即查询
    
    可以通过表单提交或JSON格式提交参数:
    - query: 查询内容
    - file: 上传的文件（可选）
    """
    try:
        # 解析请求参数
        content_type = request.headers.get("Content-Type", "")
        
        # 处理JSON请求
        if "application/json" in content_type:
            data = await request.json()
            query = data.get("query")
            file = None  # JSON请求不支持文件上传
            collection_name = data.get("collection_name", "temp_collection")
            
            if not query:
                raise HTTPException(
                    status_code=422, 
                    detail="JSON请求必须包含 query 字段"
                )
        # 处理表单请求
        elif "multipart/form-data" in content_type:
            form = await request.form()
            query = form.get("query")
            file = form.get("file")
            collection_name = form.get("collection_name", "temp_collection")
            
            if not query:
                raise HTTPException(
                    status_code=422, 
                    detail="表单请求必须包含 query 字段"
                )
        else:
            # 处理普通表单请求
            form = await request.form()
            query = form.get("query")
            file = form.get("file")
            collection_name = form.get("collection_name", "temp_collection")
            
            if not query:
                raise HTTPException(
                    status_code=422, 
                    detail="请求必须包含 query 字段"
                )
        
        # 处理文件上传
        if file and hasattr(file, "read"):
            file_content = await file.read()
            await llama_index_service.upload_file(
                file_content=file_content,
                filename=getattr(file, "filename", "uploaded_file.txt"),
                collection_name=collection_name
            )
        
        # 执行查询
        result = await llama_index_service.query(
            collection_name=collection_name,
            query_text=query
        )
        
        # 添加后台任务清理资源
        async def cleanup_resources():
            try:
                llama_index_service.cleanup()
            except Exception as e:
                logger.error(f"清理资源失败: {str(e)}")
        
        background_tasks.add_task(cleanup_resources)
        
        return result
    except Exception as e:
        logger.error(f"查询失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

@router.post("/test", tags=["RAG-LlamaIndex"])
async def test_llama_index(
    query: str = Form(...),
    file: Optional[UploadFile] = File(None),
    background_tasks: BackgroundTasks = None,
    llama_index_service: LlamaIndexRAGService = Depends(get_llama_index_service)
):
    """
    测试 LlamaIndex RAG 功能
    
    参数:
    - query: 查询文本
    - file: 上传的文件（可选）
    """
    try:
        collection_name = "test_collection"
        
        # 处理文件上传
        if file:
            file_content = await file.read()
            await llama_index_service.upload_file(
                file_content=file_content,
                filename=file.filename,
                collection_name=collection_name
            )
        
        # 执行查询
        result = await llama_index_service.query(
            collection_name=collection_name,
            query_text=query
        )
        
        # 添加后台任务清理资源
        if background_tasks:
            async def cleanup_resources():
                try:
                    llama_index_service.cleanup()
                except Exception as e:
                    logger.error(f"清理资源失败: {str(e)}")
            
            background_tasks.add_task(cleanup_resources)
        
        return result
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"测试失败: {str(e)}")
