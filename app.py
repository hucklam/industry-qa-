import os
import sys

# 添加目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import uuid

from services.parser import DocumentParser
from services.vector_store import VectorStore
from services.llm import LLMService

app = Flask(__name__)
CORS(app)

# 初始化服务
vector_store = VectorStore()
llm_service = LLMService()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/upload", methods=["POST"])
def upload_document():
    """上传文档并入库"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    ext = file.filename.lower().split('.')[-1]
    if ext not in DocumentParser.SUPPORTED_TYPES:
        return jsonify({"error": f"不支持的文件类型: {ext}"}), 400

    try:
        content = file.read()
        text = DocumentParser.parse(content, file.filename)
        chunks = DocumentParser.chunk_text(text)

        doc_id = str(uuid.uuid4())
        vector_store.add_documents(doc_id, chunks, file.filename)

        return jsonify({
            "success": True,
            "message": f"文档 {file.filename} 已成功入库",
            "doc_id": doc_id,
            "chunks": len(chunks)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    """问答接口"""
    data = request.get_json()
    question = data.get("question", "")

    # 1. 检索相关文档
    context = vector_store.similarity_search(question, top_k=3)

    # 2. 检查相关性
    if not llm_service.check_relevance(question, context):
        return jsonify({
            "answer": "抱歉，您的问题超出了当前知识库的范围，无法回答。",
            "sources": []
        })

    # 3. 生成答案
    result = llm_service.generate_answer(question, context)

    return jsonify({
        "answer": result["answer"],
        "sources": result["sources"]
    })

# Vercel 入口
def handler(environ, start_response):
    return app(environ, start_response)
