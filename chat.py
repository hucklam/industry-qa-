import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.vector_store import VectorStore
from services.llm import LLMService

# 全局变量
_vector_store = None
_llm_service = None

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store

def get_llm_service():
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service

def main(request):
    if request.method == "POST":
        try:
            data = request.get_json()
            question = data.get("question", "")

            if not question:
                return {"error": "Question is required"}, 400

            vector_store = get_vector_store()
            llm_service = get_llm_service()

            # 1. 检索相关文档
            context = vector_store.similarity_search(question, top_k=3)

            # 2. 检查相关性
            if not llm_service.check_relevance(question, context):
                return {
                    "answer": "抱歉，您的问题超出了当前知识库的范围，无法回答。",
                    "sources": []
                }

            # 3. 生成答案
            result = llm_service.generate_answer(question, context)

            return {
                "answer": result["answer"],
                "sources": result["sources"]
            }

        except Exception as e:
            return {"error": str(e)}, 500

    return {"error": "Method not allowed"}, 405
