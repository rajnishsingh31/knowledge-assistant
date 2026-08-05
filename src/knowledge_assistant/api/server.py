import uvicorn


def main() -> None:
    uvicorn.run(
        "knowledge_assistant.api.app:app",
        host="127.0.0.1",
        port=8000,
    )