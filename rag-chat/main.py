from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from rag.fake_chain import get_rag_chain
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



rag_chain = None


class Question(BaseModel):
    question: str


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(data: Question):
    global rag_chain

    if rag_chain is None:
        rag_chain = get_rag_chain()

    result = rag_chain(data.question)
    return {"answer": result}