from langchain_community.document_loaders import GutenbergLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_docs():
    """
    :return: book a list of pages from Moby Dick
    """
    loader = GutenbergLoader("https://www.gutenberg.org/cache/epub/2701/pg2701.txt")
    book = loader.load()
    return book


def split_docs(blog_docs):
    """
    :param blog_docs:
    :return:
    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300,
        chunk_overlap=50)

    # Make splits
    splits = text_splitter.split_documents(blog_docs)
    return splits


def get_retriever(blog_docs) -> OpenAIEmbeddings:
    """

    :return:
    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300,
        chunk_overlap=50)

    # Make splits
    splits = text_splitter.split_documents(blog_docs)

    vectorstore = FAISS.from_documents(documents=splits,
                                       embedding=OpenAIEmbeddings())

    retriever = vectorstore.as_retriever()

    return retriever
