# RAG Demo

基于LlamaIndex的检索增强生成(RAG)演示项目。

## 当前进度

1. **基础RAG功能** - 已完成
   - 文档上传和处理
   - 向量索引和存储
   - 基于语义的文档检索
   - 集合管理

2. **CSV问答对处理** - 已完成
   - 支持CSV格式的问答对文件
   - 每个问答对作为独立的知识库单元
   - 保留原始问题和答案作为元数据
   - 支持精确的问答检索

## 功能特点

- 文档上传和处理
- 向量索引和存储
- 基于语义的文档检索
- 集合管理
- 支持多种文件格式
- 特殊处理CSV问答对文件

## 技术栈

- FastAPI: Web框架
- LlamaIndex: RAG核心框架
- ChromaDB: 向量数据库
- HuggingFace: 嵌入模型
- OpenAI/OpenRouter: LLM服务

## 安装

1. 克隆仓库:

```bash
git clone <repository-url>
cd RAG_demo
```

2. 安装依赖:

```bash
pip install -r requirements.txt
```

3. 创建`.env`文件并设置必要的环境变量:

```
OPENAI_API_KEY=your_api_key
```

## 运行

```bash
python run.py
```

应用将在 http://localhost:8000 上运行。

## API端点

### 上传文件到集合

```
POST /api/v1/rag/llama-index/collections/{collection_name}/upload
```

支持的文件格式：
- 文本文件（.txt）
- CSV问答对文件（.csv）- 每行作为一个独立的知识库单元
- 其他LlamaIndex支持的文档格式

### 查询集合

```
POST /api/v1/rag/llama-index/query
```

请求体:
```json
{
  "collection_name": "your_collection",
  "query": "your query text",
  "top_k": 3
}
```

### 上传文件并立即查询

```
POST /api/v1/rag/llama-index/query-with-file
```

### 测试功能

```
POST /api/v1/rag/llama-index/test
```

## 使用示例

### 上传CSV问答对文件

```bash
curl -X POST -F "file=@qasample.csv" http://localhost:8000/api/v1/rag/llama-index/collections/qa_collection/upload
```

### 查询问答对

```bash
curl -X POST -H "Content-Type: application/json" -d '{"collection_name": "qa_collection", "query": "Can cats eat chocolate?", "top_k": 3}' http://localhost:8000/api/v1/rag/llama-index/query
```

## 目录结构

```
RAG_demo/
├── app/
│   ├── ai/
│   │   └── llama_index_rag.py  # RAG核心实现
│   ├── api/
│   │   ├── router.py           # API路由注册
│   │   └── routes/
│   │       └── rag/
│   │           └── llama_index.py  # RAG API端点
│   └── core/
│       └── config.py           # 应用配置
├── huggingface/                # 嵌入模型文件
├── chroma_db/                  # 向量数据库存储
├── .env                        # 环境变量
├── main.py                     # 应用入口
├── run.py                      # 运行脚本
├── test_data.txt               # 测试文本文件
├── qasample.csv                # 测试问答对文件
└── requirements.txt            # 依赖列表
```
