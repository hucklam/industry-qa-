import json
import os
import sys
import uuid

# 添加目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.parser import DocumentParser
from services.vector_store import VectorStore

# 全局变量（Serverless 环境）
_vector_store = None

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store

def main(request):
    if request.method == "GET":
        return {"status": "ok"}

    if request.method == "POST":
        try:
            # 解析 multipart form data
            content_type = request.headers.get("content-type", "")

            if "multipart/form-data" in content_type:
                # 获取文件
                file = request.files.get("file")
                if not file:
                    return {"error": "No file provided"}, 400

                ext = file.filename.lower().split('.')[-1]
                if ext not in DocumentParser.SUPPORTED_TYPES:
                    return {"error": f"不支持的文件类型: {ext}"}, 400

                file_content = file.read()
                text = DocumentParser.parse(file_content, file.filename)
                chunks = DocumentParser.chunk_text(text)

                doc_id = str(uuid.uuid4())
                vector_store = get_vector_store()
                vector_store.add_documents(doc_id, chunks, file.filename)

                return {
                    "success": True,
                    "message": f"文档 {file.filename} 已成功入库",
                    "doc_id": doc_id,
                    "chunks": len(chunks)
                }
            else:
                return {"error": "Invalid content type"}, 400

        except Exception as e:
            return {"error": str(e)}, 500

    return {"error": "Method not allowed"}, 405
