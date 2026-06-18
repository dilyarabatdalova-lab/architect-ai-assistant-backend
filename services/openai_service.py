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
            self.client = OpenAI(api_key=api_key, timeout=20.0)

        if provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
            if not api_key:
                return

            self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
            self.client = OpenAI(
                api_key=api_key,
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                timeout=20.0,
            )

    def is_ready(self) -> bool:
        return self.client is not None and bool(self.model)

    def answer(self, question: str, chunks: List[Dict[str, str]]) -> str:
        if not self.is_ready():
            return ""

        context = self._format_context(chunks)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                max_tokens=450,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ты ИИ-помощник архитектора внутри обучающего приложения. "
                            "Отвечай на русском языке и только на основании переданного контекста. "
                            "Если ответа в контексте нет, напиши строго: "
                            f"{NO_INFO_ANSWER} "
                            "Не придумывай нормы, цифры, названия документов и разделы. "
                            "Ответ должен быть самодостаточным для мобильного чата. "
                            "Не отправляй пользователя смотреть таблицу, раздел, приложение, чертёж или источник, "
                            "если сами данные не приведены в контексте. "
                            "Если в контексте сказано только, что значения указаны в таблице, "
                            "но самих значений нет, напиши, что точные значения в базе знаний не раскрыты."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Вопрос пользователя:\n{question}\n\n"
                            f"Фрагменты базы знаний:\n{context}\n\n"
                            "Сформируй краткий понятный ответ без отдельного блока источников."
                        ),
                    },
                ],
            )
        except Exception:
            return ""

        return response.choices[0].message.content or ""

    def _format_context(self, chunks: List[Dict[str, str]]) -> str:
        parts = []

        for index, chunk in enumerate(chunks, start=1):
            parts.append(
                f"[{index}] Источник: {chunk['source']}\n"
                f"{chunk['text']}"
            )

        return "\n\n".join(parts)
