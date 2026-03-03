from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import uuid
import os

from services.parser import DocumentParser
from services.vector_store import VectorStore
from services.llm import LLMService

# 加载环境变量
load_dotenv()

app = FastAPI(title="行业知识问答系统")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化服务
vector_store = VectorStore()
llm_service = LLMService()

# 请求模型
class ChatRequest(BaseModel):
    question: str
    system_prompt: str = "你是一个专业的知识库问答助手。"

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传文档并入库"""
    # 检查文件类型
    ext = file.filename.lower().split('.')[-1]
    if ext not in DocumentParser.SUPPORTED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型。支持: {DocumentParser.SUPPORTED_TYPES}"
        )

    # 读取内容
    content = await file.read()

    # 解析文档
    try:
        text = DocumentParser.parse(content, file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 分块
    chunks = DocumentParser.chunk_text(text)

    # 生成 ID 并入库
    doc_id = str(uuid.uuid4())
    vector_store.add_documents(doc_id, chunks, file.filename)

    return {
        "success": True,
        "message": f"文档 {file.filename} 已成功入库",
        "doc_id": doc_id,
        "chunks": len(chunks)
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """问答接口"""

    # 1. 检索相关文档
    context = vector_store.similarity_search(request.question, top_k=3)

    # 2. 检查相关性
    if not llm_service.check_relevance(request.question, context):
        return ChatResponse(
            answer="抱歉，您的问题超出了当前知识库的范围，无法回答。",
            sources=[]
        )

    # 3. 生成答案
    result = llm_service.generate_answer(request.question, context)

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"]
    )

# 启动命令: uvicorn main:app --reload --port 8000
