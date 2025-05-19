from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
from uuid import uuid4

from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_docs"
VECTOR_DIR = "faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/Mistral-7B-v0.1"  # Or use phi-2

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# Load or initialize vector DB
if os.path.exists(VECTOR_DIR):
    vector_db = FAISS.load_local(VECTOR_DIR, embedding_model, allow_dangerous_deserialization=True)
else:
    vector_db = None

# Load LLM
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, torch_dtype="auto")
llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=512)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

class Question(BaseModel):
    question: str

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, f"{uuid4()}_{file.filename}")
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    if file.filename.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file.filename.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        return {"error": "Unsupported file format"}

    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    global vector_db
    if vector_db:
        vector_db.add_documents(docs)
    else:
        vector_db = FAISS.from_documents(docs, embedding_model)

    vector_db.save_local(VECTOR_DIR)
    return {"message": "File uploaded and processed successfully."}

@app.post("/ask")
def ask_question(data: Question):
    if not vector_db:
        return {"error": "No training data available yet."}

    retriever = vector_db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = qa_chain.run(data.question)
    return {"answer": result}
