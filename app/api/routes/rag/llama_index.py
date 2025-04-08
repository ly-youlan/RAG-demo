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

class QueryResponse(BaseModel):
    """查询响应模型"""
    answer: str
    sources: List[Dict[str, Any]]

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
        # 读取文件内容
        file_content = await file.read()
        
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
                except Exception as e:
                    logger.error(f"清理资源失败: {str(e)}")
            
            background_tasks.add_task(cleanup_resources)
        
        return result
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
