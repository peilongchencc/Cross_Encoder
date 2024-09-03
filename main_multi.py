"""
Description: 多进程版本主程序。
Notes: 
"""
import uvicorn
from loguru import logger
from fastapi import FastAPI
from pydantic import BaseModel
from rank_documents import CustomCrossEncoder
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import torch.multiprocessing as mp

bge_rank_model = None  # 声明一个全局变量以供在各个路由中使用

@asynccontextmanager
async def lifespan(app: FastAPI):
    """程序启动前启动多进程和模型"""
    global bge_rank_model
    mp.set_start_method('spawn', force=True)  # 在应用启动时设置多进程启动方式
    bge_rank_model = CustomCrossEncoder()  # 初始化模型
    logger.info("bge模型完成加载")
    yield

app = FastAPI(lifespan=lifespan)

# 设置允许前端跨域连接
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # 允许所有的源
    allow_credentials=True,
    allow_methods=["*"],    # 允许所有的HTTP方法
    allow_headers=["*"],    # 允许所有的HTTP头
)

class QueryRequest(BaseModel):
    query: str
    documents: list[str]

# 根目录访问的处理
@app.get("/")
async def read_root():
    return {"code": 0, "msg": "欢迎访问CrossEncoder在线测试", "data": ""}

@app.post("/rank_documents/")
async def rank_documents_endpoint(request: QueryRequest):
    ranked_documents = bge_rank_model.rank_documents(request.query, request.documents, num_workers=4)  # 使用4个子进程
    return {"ranked_documents": ranked_documents}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8800)