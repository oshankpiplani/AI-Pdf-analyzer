# PDF Q&A API

A FastAPI-based REST API that allows users to upload PDF documents and ask questions about their content using AI-powered question answering. The API supports both OpenAI GPT-3.5-turbo and Hugging Face models for natural language processing.

## Features

- PDF Upload & Processing: Upload PDF files and extract text content
- AI-Powered Q&A: Ask questions about uploaded documents using OpenAI or Hugging Face models
- Cloud Storage: Automatic file storage to AWS S3
- Database Integration: PostgreSQL database for document and chat history management
- Chat History: Track questions and answers for each document
- Multiple AI Backends: Supports both OpenAI GPT-3.5-turbo and Hugging Face DistilBERT
- RESTful API: Clean, documented API endpoints
- CORS Support: Cross-origin resource sharing enabled

## Technology Stack

- Backend: FastAPI (Python)
- Database: PostgreSQL with SQLAlchemy ORM
- AI/ML: OpenAI GPT-3.5-turbo, Hugging Face Transformers
- Cloud Storage: AWS S3
- PDF Processing: PyPDF2
- Authentication: Environment-based configuration

## Prerequisites

- Python 3.8+
- PostgreSQL database
- AWS S3 bucket (optional but recommended)
- OpenAI API key (optional, falls back to Hugging Face)

## Installation



1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables
   
   Create a `.env` file in the root directory:
   ```env
   # Database
   DATABASE_URL=postgresql://username:password@localhost:5432/pdfqa
   
   # AWS S3 (Required)
   AWS_ACCESS_KEY_ID=your_aws_access_key
   AWS_SECRET_ACCESS_KEY=your_aws_secret_key
   AWS_REGION=us-east-1
   S3_BUCKET_NAME=your-s3-bucket-name
   
   # OpenAI (Optional - falls back to Hugging Face if not provided)
   OPENAI_API_KEY=your_openai_api_key
   ```

3. Set up the database
   
   Create a PostgreSQL database and update the `DATABASE_URL` in your `.env` file. The application will automatically create the required tables on startup.

4. Run the application
   ```bash
   python main.py
   ```
 