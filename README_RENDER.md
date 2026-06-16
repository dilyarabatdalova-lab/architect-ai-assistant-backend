# Загрузка backend на Render

Эта папка содержит сервер для модуля "ИИ-помощник архитектора".

## Что загружать

На Render/GitHub нужно загружать содержимое папки `backend`:

- `main.py`
- `requirements.txt`
- `runtime.txt`
- `render.yaml`
- папку `services`
- папку `documents`

Не нужно загружать:

- `.venv`
- `__pycache__`
- `vector_db`
- `uploads`
- `.env`

Эти папки уже добавлены в `.gitignore` и `.renderignore`.

## Настройки Render

При создании Web Service укажи:

- Language: `Python 3`
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Environment Variable:
  - Key: `AI_PROVIDER`
  - Value: `none`

В режиме `AI_PROVIDER=none` сервер работает бесплатно и отвечает только по подготовленным документам из папки `documents`.

## Проверка после загрузки

После успешного деплоя Render выдаст адрес вида:

`https://architect-ai-assistant.onrender.com`

Проверь в браузере:

`https://architect-ai-assistant.onrender.com/`

Должен открыться ответ:

```json
{
  "status": "ok",
  "message": "Architect AI Assistant backend is running"
}
```

Адрес для Unity:

`https://architect-ai-assistant.onrender.com/chat`

Именно его нужно вставить в `ApiClient.apiUrl`.

## Если позже нужен DeepSeek

В Render можно поменять переменные:

- `AI_PROVIDER=deepseek`
- `DEEPSEEK_API_KEY=твой_ключ`
- `DEEPSEEK_MODEL=deepseek-chat`
- `DEEPSEEK_BASE_URL=https://api.deepseek.com`

После изменения переменных нужно нажать redeploy.
