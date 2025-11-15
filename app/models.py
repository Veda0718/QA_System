from pydantic import BaseModel

class QAResponse(BaseModel):
    answer: str