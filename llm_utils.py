import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

LLM_REPO_ID = "HuggingFaceH4/zephyr-7b-beta"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_llm():
    
    return HuggingFaceEndpoint(
        repo_id=LLM_REPO_ID,
        huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        max_new_tokens=900,
        temperature=0.1,
    )


def get_embedding_model():
    # Loads a small local embedding model, so no API key is needed for embeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def create_faiss_index(chunks, embedding_model):
    # Builds a searchable FAISS vector index out of the transcript chunks
    return FAISS.from_texts(chunks, embedding_model)


def retrieve(query, faiss_index, k=7):
    # Returns the k transcript chunks most relevant to the query
    return faiss_index.similarity_search(query, k=k)


def create_summary_prompt():
    # Builds the instruction template used to summarize a transcript
    template = """You are an AI assistant tasked with summarizing YouTube video transcripts.
Summarize the transcript below in a single concise paragraph. Ignore timestamps and focus only on the spoken content.

Transcript:
{transcript}

Summary:"""
    return PromptTemplate(input_variables=["transcript"], template=template)


def create_qa_prompt_template():
    # Builds the instruction template used to answer questions from retrieved context
    template = """You are an expert assistant answering questions about a video using the context below.
Be precise, well-organized, and stick to the information given.

Context:
{context}

Question:
{question}

Answer:"""
    return PromptTemplate(input_variables=["context", "question"], template=template)


def create_summary_chain(llm):
    # Builds a prompt-to-LLM pipeline (LCEL) for summarization
    return create_summary_prompt() | llm


def create_qa_chain(llm):
    # Builds a prompt-to-LLM pipeline (LCEL) for question answering
    return create_qa_prompt_template() | llm


def generate_answer(question, faiss_index, qa_chain, k=7):
    # Retrieves relevant context then asks the QA chain to answer the question
    context = retrieve(question, faiss_index, k=k)
    return qa_chain.invoke({"context": context, "question": question})
