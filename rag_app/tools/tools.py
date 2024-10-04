from operator import itemgetter
import sys

sys.path.append("../utils")
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.load import (dumps,
                            loads)
import sys
from pathlib import Path

p = Path(__file__).parents[1]
sys.path.append(f"{p}/rag_app")
from utils.utils import (get_docs,
                         split_docs,
                         get_retriever)

from dotenv import load_dotenv

load_dotenv("../env")


def get_rag_fusion_chain(retriever: OpenAIEmbeddings):
    """

    :param retriever:
    :return:
    """
    template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
    Generate multiple search queries related to: {question} \n
    Output (4 queries):"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)
    generate_queries = (
            prompt_rag_fusion
            | ChatOpenAI(temperature=0)
            | StrOutputParser()
            | (lambda x: x.split("\n"))
    )

    chain = generate_queries | retriever.map() | reciprocal_rank_fusion
    return chain


def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents
        and an optional parameter k used in the RRF formula
    """

    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results


def get_final_rag_chain(chain):
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

    return final_rag_chain
