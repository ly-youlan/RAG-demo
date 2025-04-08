"""
API 路由注册

该模块负责注册所有API路由。
"""

from fastapi import APIRouter

from app.api.routes.rag.llama_index import router as llama_index_router

# 创建主路由
api_router = APIRouter()

# 注册RAG路由
api_router.include_router(llama_index_router)
