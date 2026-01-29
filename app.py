import os
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv() # Load the environmental vairables found in .env

app = FastAPI()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # This api will look for the OPENAI_API_KEY env var.

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": "langchain-agent",
                "object": "model",
                "owned_by": "you",
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    user_msg = req.messages[-1].content

    response = llm.invoke(user_msg)

    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.content,
                },
                "finish_reason": "stop",
            }
        ],
    }
