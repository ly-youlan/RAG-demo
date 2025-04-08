"""
RAG Demo 应用程序

该模块是应用程序的入口点，负责创建FastAPI应用并注册路由。
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# 创建FastAPI应用
app = FastAPI(
    title="RAG Demo API",
    description="基于LlamaIndex的RAG演示API",
    version="1.0.0",
)

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
