
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

def load_and_split_pdf(pdf_path, chunk_size=1000, chunk_overlap=200):
    loader = PyMuPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.split_documents(pages)
    return docs

def create_vector_db(docs, api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.from_documents(docs, embeddings)
    return db

def query_pdf(db, api_key, question):
    llm = OpenAI(temperature=0, openai_api_key=api_key)
    chain = load_qa_chain(llm, chain_type="stuff")
    relevant_docs = db.similarity_search(question)
    answer = chain.run(input_documents=relevant_docs, question=question)
    return answer, relevant_docs
