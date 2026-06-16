from pathlib import Path
from typing import Dict, List
import os
import re


NO_INFO_ANSWER = "Информация по данному вопросу в базе знаний отсутствует."


class VectorService:
    """Searches prepared document fragments.

    For the Render demo the default mode is a lightweight local text search.
    It does not require ChromaDB, OpenAI embeddings, or an API key, so the
    server starts reliably on a free web service.
    """

    def __init__(self, vector_db_dir: Path, use_openai: bool) -> None:
        self.use_openai = use_openai
        self.fallback_chunks: List[Dict[str, str]] = []
        self.openai_client = None
        self.client = None
        self.collection = None

        if not self.use_openai:
            return

        try:
            import chromadb
            from openai import OpenAI
        except ImportError:
            self.use_openai = False
            return

        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.client = chromadb.PersistentClient(path=str(vector_db_dir))
        self.collection = self.client.get_or_create_collection(
            name="architect_knowledge_base"
        )

    def has_index(self) -> bool:
        return self.collection is not None and self.collection.count() > 0

    def add_chunks(self, chunks: List[Dict[str, str]]) -> None:
        if self.collection is None:
            return

        ids = [chunk["id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        metadata = [{"source": chunk["source"]} for chunk in chunks]
        embeddings = self._create_embeddings(texts)

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadata,
            embeddings=embeddings,
        )

    def set_fallback_chunks(self, chunks: List[Dict[str, str]]) -> None:
        self.fallback_chunks = chunks

    def search(self, question: str, limit: int = 5) -> List[Dict[str, str]]:
        if not self.use_openai:
            return self._fallback_search(question, limit)

        if self.collection is None or self.collection.count() == 0:
            return []

        question_embedding = self._create_embeddings([question])[0]
        result = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=limit,
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]

        chunks = []

        for text, metadata in zip(documents, metadatas):
            if not text:
                continue

            chunks.append(
                {
                    "text": text,
                    "source": metadata.get("source", "Неизвестный источник"),
                }
            )

        return chunks

    def build_fallback_answer(self, question: str, chunks: List[Dict[str, str]]) -> str:
        if not chunks:
            return NO_INFO_ANSWER

        question_words = self._tokenize(question)
        best_text = chunks[0]["text"]

        asks_about_questions = any(word.startswith("вопрос") for word in question_words)

        if "рекомендуемые вопросы клиенту:" in best_text.lower() and asks_about_questions:
            list_start = best_text.lower().find("рекомендуемые вопросы клиенту:")

            if list_start >= 0:
                answer = best_text[list_start:].strip()
                return self._trim_answer(answer)

        sentences = re.split(r"(?<=[.!?])\s+", best_text)
        scored_sentences = []

        for index, sentence in enumerate(sentences):
            sentence_words = self._tokenize(sentence)
            score = len(question_words.intersection(sentence_words))

            if score > 0:
                scored_sentences.append((score, index, sentence.strip()))

        if scored_sentences:
            scored_sentences.sort(key=lambda item: (-item[0], item[1]))
            selected = sorted(scored_sentences[:3], key=lambda item: item[1])
            answer = " ".join(sentence for _, _, sentence in selected)
        else:
            answer = best_text

        return self._trim_answer(answer)

    def _create_embeddings(self, texts: List[str]) -> List[List[float]]:
        if self.openai_client is None:
            return []

        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )

        return [item.embedding for item in response.data]

    def _fallback_search(self, question: str, limit: int) -> List[Dict[str, str]]:
        question_words = self._tokenize(question)

        if not question_words:
            return []

        scored_chunks = []

        for chunk in self.fallback_chunks:
            chunk_words = self._tokenize(chunk["text"])
            score = len(question_words.intersection(chunk_words))

            if score > 0:
                scored_chunks.append((score, chunk))

        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:limit]]

    def _tokenize(self, text: str) -> set:
        words = re.findall(r"[а-яА-ЯёЁa-zA-Z0-9]+", text.lower())
        stop_words = {
            "и", "в", "во", "на", "с", "со", "для", "по", "к", "от", "о", "об",
            "что", "какие", "какая", "какой", "необходимо", "нужно", "можно",
            "the", "a", "an", "of", "to", "in", "and",
        }

        return {word for word in words if word not in stop_words and len(word) > 2}

    def _trim_answer(self, answer: str, max_length: int = 650) -> str:
        if len(answer) > max_length:
            answer = answer[:max_length].rsplit(" ", 1)[0] + "..."

        return answer.strip() or NO_INFO_ANSWER
