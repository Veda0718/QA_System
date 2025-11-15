from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from .models import QAResponse
from .qa import fetch_messages, route

app = FastAPI(
    title="Aurora Member QA",
    description="LLM-powered question answering over Aurora messages.",
    version="1.0.0",
)

@app.get("/ask", response_model=QAResponse)
def ask(q: str = Query(..., description="Natural-language question")):
    try:
        messages = fetch_messages(limit=300)
        answer = route(q, messages)
        return {"answer": answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"answer": f"Error: {e}"})