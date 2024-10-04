from operator import itemgetter

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from tools.tools import get_rag_fusion_chain
from utils.utils import (get_docs,
                         split_docs,
                         get_retriever)

load_dotenv("env")

if __name__ == "__main__":
    blog_docs = get_docs()
    retriever = get_retriever(
                              blog_docs=blog_docs)

    # get similar docs
    # docs = retriever.invoke("What is Task Decomposition?")
    # print(docs)
    chain = get_rag_fusion_chain(retriever=retriever)

    # we can now use the context from the retrieval chain
    template = """Answer the following question based on this context:
    {context}
    Question: {question}
    
    If you do not know the answer say you don't know.
    """
    prompt = ChatPromptTemplate.from_template(template=template)
    llm = ChatOpenAI(temperature=0)

    final_rag_chain = (
            {"context": chain,
             "question": itemgetter("question")}
            | prompt
            | llm
            | StrOutputParser()
    )
    res = final_rag_chain.invoke({"question": "Why did Moby Dick hate Captain Ahab"}) # Who is Moby Dick?
    print(res)
