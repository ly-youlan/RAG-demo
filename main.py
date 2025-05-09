"""
RAG Demo 应用程序

该模块是应用程序的入口点，负责创建FastAPI应用并注册路由。
"""

import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.api.router import api_router

# 自定义中间件，用于处理大文件上传
class LargeFileHandlingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        # 增加请求体大小限制
        # 默认情况下，FastAPI限制请求体大小为1MB
        # 这里我们将其增加到200MB
        request._body_size_limit = 200 * 1024 * 1024  # 200MB
        response = await call_next(request)
        return response

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

# 设置特定模块的日志级别
logging.getLogger('app.ai.llama_index_rag').setLevel(logging.INFO)
logging.getLogger('uvicorn').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

# 创建FastAPI应用
app = FastAPI(
    title="RAG Demo API",
    description="基于LlamaIndex的RAG演示API",
    version="1.0.0",
)

# 添加大文件处理中间件
app.add_middleware(LargeFileHandlingMiddleware)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(api_router)

# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "ok"}
