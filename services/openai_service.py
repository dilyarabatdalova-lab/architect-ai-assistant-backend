import os
from typing import Dict, List, Optional


NO_INFO_ANSWER = "Информация по данному вопросу в базе знаний отсутствует."


class OpenAIService:
    """Creates final answers through the selected external AI provider.

    The OpenAI Python package can work with OpenAI-compatible APIs. DeepSeek
    supports this format, so the same code can call either OpenAI or DeepSeek.
    """

    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.client: Optional[object] = None
        self.model = ""

        try:
            from openai import OpenAI
        except ImportError:
            return

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                return

            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            self.client = OpenAI(api_key=api_key)

        if provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
            if not api_key:
                return

            self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
            self.client = OpenAI(
                api_key=api_key,
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            )

    def is_ready(self) -> bool:
        return self.client is not None and bool(self.model)

    def answer(self, question: str, chunks: List[Dict[str, str]]) -> str:
        if not self.is_ready():
            return ""

        context = self._format_context(chunks)

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты ИИ-помощник архитектора внутри обучающего приложения. "
                        "Отвечай только на основании переданного контекста. "
                        f"Если ответа в контексте нет, напиши строго: {NO_INFO_ANSWER} "
                        "Не придумывай нормы, цифры, названия документов и разделы."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Вопрос пользователя:\n{question}\n\n"
                        f"Фрагменты базы знаний:\n{context}\n\n"
                        "Сформируй краткий понятный ответ. Не добавляй блок источников."
                    ),
                },
            ],
        )

        return response.choices[0].message.content or ""

    def _format_context(self, chunks: List[Dict[str, str]]) -> str:
        parts = []

        for index, chunk in enumerate(chunks, start=1):
            parts.append(
                f"[{index}] Источник: {chunk['source']}\n"
                f"{chunk['text']}"
            )

        return "\n\n".join(parts)
