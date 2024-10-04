from query import Query
from fastapi import FastAPI
from utils.utils import (get_retriever,
                         split_docs,
                         get_docs)
from tools.tools import (get_rag_fusion_chain,
                         get_final_rag_chain)

app = FastAPI()
blog_docs = get_docs()
retriever = get_retriever(blog_docs=blog_docs)

rag_fusion_chain = get_rag_fusion_chain(retriever)
CHAIN = get_final_rag_chain(rag_fusion_chain)


@app.post("/query")
async def predict(query: Query):
    response = CHAIN.invoke({"question": query.query})
    return {"response": response}
