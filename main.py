from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import get_rag_chain

app = FastAPI()
qa_chain = get_rag_chain()

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
def chat(request: QueryRequest):
    response = qa_chain.run(request.query)
    return {"response": response}
