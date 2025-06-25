from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
import PyPDF2
import io
import os
from typing import Optional
from openai import OpenAI  
from transformers import pipeline
import re
import logging

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/pdfqa")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI
app = FastAPI(title="PDF Q&A API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    s3_key = Column(String)
    content = Column(Text)
    upload_date = Column(DateTime, default=datetime.utcnow)
    is_processed = Column(Boolean, default=False)

class ChatHistory(Base):
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer)
    question = Column(Text)
    answer = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    document_id: Optional[int] = None

class QuestionResponse(BaseModel):
    answer: str
    document_id: Optional[int] = None
    ai_source: str  
    confidence: Optional[float] = None  

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize services
s3_client = None
qa_pipeline = None
openai_client = None

def initialize_services():
    global s3_client, qa_pipeline, openai_client
    
    # Initialize S3
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
    
    # Initialize OpenAI or Hugging Face
    if OPENAI_API_KEY:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("✅ PRIMARY AI SERVICE: OpenAI GPT-3.5-turbo initialized")
    else:
        try:
            qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                return_tensors="pt"
            )
            logger.info("✅ PRIMARY AI SERVICE: Hugging Face DistilBERT initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Hugging Face pipeline: {e}")

initialize_services()

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")

def upload_to_s3(file_content: bytes, filename: str) -> str:
    """Upload file to S3 and return the key"""
    if not s3_client:
        raise HTTPException(status_code=500, detail="S3 client not configured")
    
    if not S3_BUCKET_NAME:
        raise HTTPException(status_code=500, detail="S3 bucket name not configured")
    
    try:
        s3_key = f"pdfs/{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{filename}"
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=file_content,
            ContentType="application/pdf"
        )
        return s3_key
    except ClientError as e:
        logger.error(f"Error uploading to S3: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

def create_enhanced_prompt(question: str, context: str) -> str:
    return f"""You are an expert document analyst. I will provide you with a question about a PDF document and the relevant content from that document. Please provide a comprehensive, accurate, and helpful answer based strictly on the information provided in the document.

Guidelines for your response:
1. Answer directly and concisely
2. Use only information from the provided document content
3. If the answer isn't in the document, clearly state that
4. Provide specific details and examples when available
5. Structure your answer clearly with bullet points or numbered lists when appropriate
6. If referencing specific sections, quote them briefly

Document Content:
{context}

Question: {question}

Please provide a detailed answer based on the document content above:"""

def answer_question_openai(question: str, context: str) -> dict:
    """Answer question using OpenAI"""
    try:
        prompt = create_enhanced_prompt(question, context)
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided document content."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        return {
            "answer": response.choices[0].message.content.strip(),
            "ai_source": "OpenAI GPT-3.5-turbo",
            "confidence": None
        }
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

def answer_question_huggingface(question: str, context: str) -> dict:
    """Answer question using Hugging Face"""
    try:
        # Truncate context if too long (BERT has token limits)
        max_context_length = 3000
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        result = qa_pipeline(question=question, context=context)
        
        # Enhance the answer with context
        answer = result['answer']
        confidence = result['score']
        
        if confidence < 0.1:
            answer = f"I couldn't find a confident answer to your question in the document. The best match I found was: {answer}, but this may not be accurate. Please try rephrasing your question or ask about content that might be more explicitly stated in the document."
        
        return {
            "answer": answer,
            "ai_source": "Hugging Face DistilBERT",
            "confidence": round(confidence, 3)
        }
    except Exception as e:
        logger.error(f"Hugging Face pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload and process PDF file"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Extract text from PDF
        text_content = extract_text_from_pdf(file_content)
        
        # Clean text content to remove NUL characters and other problematic characters
        # This is crucial for PostgreSQL compatibility
        text_content = text_content.replace('\x00', '')  # Remove NUL characters
        text_content = ''.join(char for char in text_content if ord(char) >= 32 or char in '\n\r\t')  # Keep only printable characters
        
        # Upload to S3
        s3_key = None
        if s3_client and S3_BUCKET_NAME:
            try:
                s3_key = upload_to_s3(file_content, file.filename)
                logger.info(f"File uploaded to S3: {s3_key}")
            except Exception as e:
                logger.error(f"S3 upload failed: {e}")
                raise HTTPException(status_code=500, detail=f"S3 upload failed: {str(e)}")
        else:
            raise HTTPException(status_code=500, detail="S3 not configured properly")
        
        # Save to database
        db_document = Document(
            filename=file.filename,
            s3_key=s3_key,
            content=text_content,
            is_processed=True
        )
        db.add(db_document)
        db.commit()
        db.refresh(db_document)
        
        return {
            "message": "PDF uploaded and processed successfully",
            "document_id": db_document.id,
            "filename": file.filename,
            "s3_key": s3_key,
            "s3_url": f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}",
            "content_preview": text_content[:200] + "..." if len(text_content) > 200 else text_content
        }
    
    except Exception as e:
        logger.error(f"Error uploading PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")

@app.post("/ask-question/", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest, db: Session = Depends(get_db)):
    """Ask a question about the uploaded document"""
    try:
        # Get the most recent document if no document_id provided
        if request.document_id:
            document = db.query(Document).filter(Document.id == request.document_id).first()
        else:
            document = db.query(Document).order_by(Document.upload_date.desc()).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="No document found")
        
        if not document.is_processed:
            raise HTTPException(status_code=400, detail="Document is still being processed")
        
        # Generate answer
        if openai_client:
            result = answer_question_openai(request.question, document.content)
        elif qa_pipeline:
            result = answer_question_huggingface(request.question, document.content)
        else:
            raise HTTPException(status_code=500, detail="No Q&A service available")
        
        # Save to chat history (including AI source info)
        chat_entry = ChatHistory(
            document_id=document.id,
            question=request.question,
            answer=f"[{result['ai_source']}] {result['answer']}"  
        )
        db.add(chat_entry)
        db.commit()
        
        return QuestionResponse(
            answer=result['answer'],
            document_id=document.id,
            ai_source=result['ai_source'],
            confidence=result['confidence']
        )
    
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.get("/documents/")
async def get_documents(db: Session = Depends(get_db)):
    """Get list of uploaded documents"""
    documents = db.query(Document).order_by(Document.upload_date.desc()).all()
    return [
        {
            "id": doc.id,
            "filename": doc.filename,
            "upload_date": doc.upload_date,
            "is_processed": doc.is_processed,
            "s3_key": doc.s3_key,
            "s3_url": f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{doc.s3_key}" if doc.s3_key else None
        }
        for doc in documents
    ]

@app.get("/chat-history/{document_id}")
async def get_chat_history(document_id: int, db: Session = Depends(get_db)):
    """Get chat history for a specific document"""
    history = db.query(ChatHistory).filter(
        ChatHistory.document_id == document_id
    ).order_by(ChatHistory.timestamp.asc()).all()
    
    return [
        {
            "id": chat.id,
            "question": chat.question,
            "answer": chat.answer,
            "timestamp": chat.timestamp
        }
        for chat in history
    ]

@app.delete("/chat-history/{document_id}")
async def delete_chat_history(document_id: int, db: Session = Depends(get_db)):
    """Delete all chat history for a specific document"""
    try:
        # Check if document exists
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete all chat history for this document
        deleted_count = db.query(ChatHistory).filter(
            ChatHistory.document_id == document_id
        ).delete()
        
        db.commit()
        
        return {
            "message": f"Successfully deleted {deleted_count} chat messages for document {document.filename}",
            "deleted_count": deleted_count,
            "document_id": document_id
        }
    
    except Exception as e:
        logger.error(f"Error deleting chat history: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting chat history: {str(e)}")

@app.delete("/chat-message/{message_id}")
async def delete_chat_message(message_id: int, db: Session = Depends(get_db)):
    """Delete a specific chat message"""
    try:
        # Find the chat message
        chat_message = db.query(ChatHistory).filter(ChatHistory.id == message_id).first()
        if not chat_message:
            raise HTTPException(status_code=404, detail="Chat message not found")
        
        # Delete the message
        db.delete(chat_message)
        db.commit()
        
        return {
            "message": "Chat message deleted successfully",
            "message_id": message_id
        }
    
    except Exception as e:
        logger.error(f"Error deleting chat message: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting chat message: {str(e)}")

@app.get("/")
async def root():
    return {"message": "PDF Q&A API is running"}

@app.get("/test-s3/")
async def test_s3():
    """Test S3 connection"""
    if not s3_client:
        return {"status": "error", "message": "S3 client not initialized"}
    
    if not S3_BUCKET_NAME:
        return {"status": "error", "message": "S3 bucket name not configured"}
    
    try:
        # Try to list objects in the bucket
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, MaxKeys=1)
        return {
            "status": "success", 
            "message": f"S3 connection successful to bucket: {S3_BUCKET_NAME}",
            "bucket_exists": True
        }
    except ClientError as e:
        return {
            "status": "error", 
            "message": f"S3 connection failed: {str(e)}",
            "bucket_name": S3_BUCKET_NAME
        }

@app.get("/ai-status/")
async def get_ai_status():
    """Get information about available AI services"""
    return {
        "openai_available": openai_client is not None,
        "huggingface_available": qa_pipeline is not None,
        "primary_service": "OpenAI GPT-3.5-turbo" if openai_client else "Hugging Face DistilBERT" if qa_pipeline else "None",
        "openai_configured": OPENAI_API_KEY is not None,
    }


# Add this new endpoint to your FastAPI backend

@app.delete("/documents/{document_id}")
async def delete_document(document_id: int, db: Session = Depends(get_db)):
    """Delete a document and all its associated chat history"""
    try:
        # Get the document
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete associated chat history first
        chat_count = db.query(ChatHistory).filter(
            ChatHistory.document_id == document_id
        ).delete()
        
        # Delete from S3 if exists
        s3_deleted = False
        if s3_client and document.s3_key:
            try:
                s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=document.s3_key)
                s3_deleted = True
                logger.info(f"Deleted S3 object: {document.s3_key}")
            except ClientError as e:
                logger.error(f"Error deleting S3 object: {e}")
                # Continue with database deletion even if S3 deletion fails
        
        # Store filename for response
        filename = document.filename
        
        # Delete from database
        db.delete(document)
        db.commit()
        
        return {
            "message": f"Successfully deleted document '{filename}' and {chat_count} associated chat messages",
            "document_id": document_id,
            "filename": filename,
            "chat_messages_deleted": chat_count,
            "s3_deleted": s3_deleted
        }
    
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

# Also add this endpoint to delete multiple documents at once (optional)
@app.delete("/documents/")
async def delete_multiple_documents(document_ids: list[int], db: Session = Depends(get_db)):
    """Delete multiple documents and their associated chat history"""
    try:
        deleted_documents = []
        total_chat_messages = 0
        s3_deleted_count = 0
        
        for document_id in document_ids:
            # Get the document
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                continue  # Skip if document doesn't exist
            
            # Delete associated chat history
            chat_count = db.query(ChatHistory).filter(
                ChatHistory.document_id == document_id
            ).delete()
            total_chat_messages += chat_count
            
            # Delete from S3 if exists
            if s3_client and document.s3_key:
                try:
                    s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=document.s3_key)
                    s3_deleted_count += 1
                except ClientError as e:
                    logger.error(f"Error deleting S3 object {document.s3_key}: {e}")
            
            # Store document info
            deleted_documents.append({
                "id": document.id,
                "filename": document.filename,
                "chat_messages_deleted": chat_count
            })
            
            # Delete from database
            db.delete(document)
        
        db.commit()
        
        return {
            "message": f"Successfully deleted {len(deleted_documents)} documents and {total_chat_messages} chat messages",
            "deleted_documents": deleted_documents,
            "total_chat_messages_deleted": total_chat_messages,
            "s3_files_deleted": s3_deleted_count
        }
    
    except Exception as e:
        logger.error(f"Error deleting multiple documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting documents: {str(e)}")
    



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)