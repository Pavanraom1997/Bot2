from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import os
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Folder to store uploaded docs
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
def chat(req: ChatRequest):
    # Dummy response logic - replace with vector/LLM lookup
    query = req.query.lower()
    if "refund" in query:
        return {"answer": "You can request a refund from the transaction section in ERP under user details."}
    elif "password" in query:
        return {"answer": "Please use the 'Forgot Password' link on the login page to reset your password."}
    else:
        return {"answer": "I'm not sure about that. Please contact support."}

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": f"Uploaded {file.filename} successfully."}
