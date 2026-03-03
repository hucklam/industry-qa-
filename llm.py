import httpx
import os
from typing import List, Dict

class LLMService:
    """LLM 服务 - 使用 MiniMax M2.5"""

    def __init__(self):
        self.api_key = os.getenv("MINIMAX_API_KEY")
        self.base_url = os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com")
        self.model = os.getenv("LLM_MODEL", "MiniMax-M2.5")

    def _call_api(self, messages: List[Dict], temperature: float = 0.3) -> str:
        """调用 MiniMax API"""
        url = f"{self.base_url}/v1/chat/completions"

        response = httpx.post(
            url,
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            },
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=60.0
        )

        if response.status_code != 200:
            raise Exception(f"API调用失败: {response.status_code} - {response.text}")

        result = response.json()
        return result["choices"][0]["message"]["content"]

    def generate_answer(
        self,
        question: str,
        context: List[Dict],
        system_prompt: str = ""
    ) -> Dict:
        """生成答案"""

        # 构建上下文
        context_text = "\n\n".join([
            f"[来源: {item['source']}]\n{item['content']}"
            for item in context
        ])

        # 构建 Prompt（强制引用来源）
        prompt = f"""你是一个专业的知识库问答助手。请根据以下参考资料回答用户的问题。

要求：
1. 必须根据提供的资料回答，不要编造信息
2. 在回答中必须注明信息来源
3. 如果资料无法回答问题，请明确说明

参考资料：
{context_text}

用户问题：{question}

请给出回答："""

        messages = [
            {"role": "system", "content": system_prompt or "你是一个专业的知识库问答助手。"},
            {"role": "user", "content": prompt}
        ]

        answer = self._call_api(messages)

        # 提取引用的来源
        sources = list(set([item['source'] for item in context]))

        return {
            "answer": answer,
            "sources": sources
        }

    def check_relevance(self, question: str, context: List[Dict]) -> bool:
        """检查问题是否与知识库相关"""
        if not context:
            return False

        total_length = sum(len(item['content']) for item in context)
        return total_length > 100
