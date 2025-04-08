"""
运行脚本

该脚本用于启动FastAPI应用程序。
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
