import json
import logging
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from src.rag import get_answer, get_answer_stream

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="Question about company policies"
    )
    history: list[HistoryMessage] = Field(default_factory=list)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value):
        if not value.strip():
            raise ValueError("Question cannot contain only whitespace")

        return value.strip()

# Structure of each source
class Source(BaseModel):
    document: str
    page: str


# Outgoing response structure
class ChatResponse(BaseModel):
    question_received: str
    answer: str
    sources: list[Source]

# Get request to check if the API is running
@app.get("/")
def home():
    return {"message": "AI Policy Assistant API is running!"}

# Complete response POST endpoint for chat
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = get_answer(
            request.question,
            [message.model_dump() for message in request.history]
        )
        return ChatResponse(
            question_received=request.question,
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        logger.exception("Error occurred while generating chat response")
        raise HTTPException(status_code=500, detail="Something went wrong while generating the answer.")

# Stream POST endpoint for chat
@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    def event_generator():
        try:
            for event in get_answer_stream(request.question,  [message.model_dump() for message in request.history]):
                event_type = event["type"]
                event_data = json.dumps(event)
                yield f"event: {event_type}\ndata: {event_data}\n\n"
        except Exception as e:
            logger.exception("Error occurred during response streaming")

            error_event = {
                "type": "error",
                "message": "Unable to generate the answer. Please try again."
            }

            yield (
                f"event: error\n"
                f"data: {json.dumps(error_event)}\n\n"
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )