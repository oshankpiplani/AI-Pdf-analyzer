import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://0.0.0.0:8000';

function App() {
  const [messages, setMessages] = useState([]);
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [currentDocument, setCurrentDocument] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deletingMessageId, setDeletingMessageId] = useState(null);
  const [deletingDocumentId, setDeletingDocumentId] = useState(null); 
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    fetchDocuments();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const fetchDocuments = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/documents/`);
      if (response.ok) {
        const docs = await response.json();
        setDocuments(docs);
        if (docs.length > 0 && !currentDocument) {
          setCurrentDocument(docs[0]);
          loadChatHistory(docs[0].id);
        }
      }
    } catch (error) {
      console.error('Error fetching documents:', error);
    }
  };

  const loadChatHistory = async (documentId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/chat-history/${documentId}`);
      if (response.ok) {
        const history = await response.json();
        const formattedMessages = history.flatMap(chat => [
          { 
            type: 'question', 
            content: chat.question, 
            timestamp: chat.timestamp,
            id: chat.id
          },
          { 
            type: 'answer', 
            content: chat.answer, 
            timestamp: chat.timestamp,
            id: chat.id
          }
        ]);
        setMessages(formattedMessages);
      }
    } catch (error) {
      console.error('Error loading chat history:', error);
    }
  };

  const handleDeleteChatHistory = async () => {
    if (!currentDocument) return;
    
    const confirmDelete = window.confirm(
      `Are you sure you want to delete all chat history for "${currentDocument.filename}"? This action cannot be undone.`
    );
    
    if (!confirmDelete) return;

    setIsDeleting(true);
    
    try {
      const response = await fetch(`${API_BASE_URL}/chat-history/${currentDocument.id}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        const result = await response.json();
        setMessages([]);
        setUploadStatus(`Chat history deleted: ${result.deleted_count} messages removed`);
        setTimeout(() => setUploadStatus(''), 3000);
      } else {
        const error = await response.json();
        setUploadStatus(`Delete failed: ${error.detail}`);
        setTimeout(() => setUploadStatus(''), 5000);
      }
    } catch (error) {
      console.error('Error deleting chat history:', error);
      setUploadStatus('Delete failed. Please try again.');
      setTimeout(() => setUploadStatus(''), 5000);
    } finally {
      setIsDeleting(false);
    }
  };

  const handleDeleteMessage = async (messageId) => {
    if (!messageId) return;
    
    const confirmDelete = window.confirm(
      'Are you sure you want to delete this message? This action cannot be undone.'
    );
    
    if (!confirmDelete) return;

    setDeletingMessageId(messageId);
    
    try {
      const response = await fetch(`${API_BASE_URL}/chat-message/${messageId}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        setMessages(prev => prev.filter(msg => msg.id !== messageId));
        setUploadStatus('Message deleted successfully');
        setTimeout(() => setUploadStatus(''), 2000);
      } else {
        const error = await response.json();
        setUploadStatus(`Delete failed: ${error.detail}`);
        setTimeout(() => setUploadStatus(''), 3000);
      }
    } catch (error) {
      console.error('Error deleting message:', error);
      setUploadStatus('Delete failed. Please try again.');
      setTimeout(() => setUploadStatus(''), 3000);
    } finally {
      setDeletingMessageId(null);
    }
  };

  
  const handleDeleteDocument = async (documentId, documentName, event) => {
    event.stopPropagation();
    
    const confirmDelete = window.confirm(
      `Are you sure you want to permanently delete "${documentName}"?\n\nThis will:\n• Delete the PDF file from storage\n• Remove all chat history\n• Cannot be undone\n\nProceed with deletion?`
    );
    
    if (!confirmDelete) return;

    setDeletingDocumentId(documentId);
    
    try {
      const response = await fetch(`${API_BASE_URL}/documents/${documentId}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        const result = await response.json();
        
        
        setDocuments(prev => prev.filter(doc => doc.id !== documentId));
        
        
        if (currentDocument && currentDocument.id === documentId) {
          const remainingDocs = documents.filter(doc => doc.id !== documentId);
          if (remainingDocs.length > 0) {
            setCurrentDocument(remainingDocs[0]);
            loadChatHistory(remainingDocs[0].id);
          } else {
            setCurrentDocument(null);
            setMessages([]);
          }
        }
        
        setUploadStatus(`Document "${result.filename}" deleted successfully (${result.chat_messages_deleted} messages removed)`);
        setTimeout(() => setUploadStatus(''), 4000);
      } else {
        const error = await response.json();
        setUploadStatus(`Delete failed: ${error.detail}`);
        setTimeout(() => setUploadStatus(''), 5000);
      }
    } catch (error) {
      console.error('Error deleting document:', error);
      setUploadStatus('Delete failed. Please try again.');
      setTimeout(() => setUploadStatus(''), 5000);
    } finally {
      setDeletingDocumentId(null);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.type.includes('pdf')) {
      setUploadStatus('Please select a PDF file.');
      return;
    }

    setIsUploading(true);
    setUploadStatus('Uploading and processing PDF...');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE_URL}/upload-pdf/`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        setUploadStatus(`Successfully uploaded: ${result.filename}`);
        setMessages([]);
        await fetchDocuments();
        
        const newDoc = {
          id: result.document_id,
          filename: result.filename,
          upload_date: new Date().toISOString(),
          is_processed: true
        };
        setCurrentDocument(newDoc);
        
        setTimeout(() => setUploadStatus(''), 3000);
      } else {
        const error = await response.json();
        setUploadStatus(`Upload failed: ${error.detail}`);
      }
    } catch (error) {
      console.error('Upload error:', error);
      setUploadStatus('Upload failed. Please try again.');
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleSubmitQuestion = async (e) => {
    e.preventDefault();
    
    if (!currentQuestion.trim()) return;
    if (!currentDocument) {
      alert('Please upload a PDF document first.');
      return;
    }

    const question = currentQuestion.trim();
    setCurrentQuestion('');
    setIsLoading(true);

    const newQuestion = {
      type: 'question',
      content: question,
      timestamp: new Date().toISOString(),
      id: null
    };
    setMessages(prev => [...prev, newQuestion]);

    try {
      const response = await fetch(`${API_BASE_URL}/ask-question/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: question,
          document_id: currentDocument.id
        }),
      });

      if (response.ok) {
        const result = await response.json();
        const newAnswer = {
          type: 'answer',
          content: result.answer,
          timestamp: new Date().toISOString(),
          id: null
        };
        setMessages(prev => [...prev, newAnswer]);
        
        setTimeout(() => loadChatHistory(currentDocument.id), 500);
      } else {
        const error = await response.json();
        const errorAnswer = {
          type: 'answer',
          content: `Sorry, I encountered an error: ${error.detail}`,
          timestamp: new Date().toISOString(),
          id: null
        };
        setMessages(prev => [...prev, errorAnswer]);
      }
    } catch (error) {
      console.error('Error asking question:', error);
      const errorAnswer = {
        type: 'answer',
        content: 'Sorry, I encountered a network error. Please try again.',
        timestamp: new Date().toISOString(),
        id: null
      };
      setMessages(prev => [...prev, errorAnswer]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDocumentSelect = (doc) => {
    setCurrentDocument(doc);
    loadChatHistory(doc.id);
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="app">
      <div className="header">
        <h1>PDF Q&A Assistant</h1>
        <div className="upload-section">
          {uploadStatus && (
            <div className={`upload-status ${uploadStatus.includes('failed') ? 'error' : 'success'}`}>
              {uploadStatus}
            </div>
          )}
          <button 
            className="upload-btn" 
            onClick={triggerFileInput}
            disabled={isUploading}
          >
            {isUploading ? 'Uploading...' : 'Upload PDF'}
          </button>
          {currentDocument && messages.length > 0 && (
            <button 
              className="delete-btn" 
              onClick={handleDeleteChatHistory}
              disabled={isDeleting || isLoading}
              title="Delete all chat history for current document"
            >
              {isDeleting ? 'Deleting...' : '🗑️ Clear All'}
            </button>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            onChange={handleFileUpload}
            style={{ display: 'none' }}
          />
        </div>
      </div>

      <div className="main-content">
        <div className="sidebar">
          <h3>Documents</h3>
          <div className="documents-list">
            {documents.length === 0 ? (
              <p className="no-documents">No documents uploaded yet</p>
            ) : (
              documents.map(doc => (
                <div 
                  key={doc.id} 
                  className={`document-item ${currentDocument?.id === doc.id ? 'active' : ''}`}
                  onClick={() => handleDocumentSelect(doc)}
                >
                  <button 
                    className="document-delete-btn"
                    onClick={(e) => handleDeleteDocument(doc.id, doc.filename, e)}
                    disabled={deletingDocumentId === doc.id}
                    title={`Delete "${doc.filename}" permanently`}
                  >
                    {deletingDocumentId === doc.id ? '⏳' : '🗑️'}
                  </button>
                  <div className="document-content">
                    <div className="document-name">{doc.filename}</div>
                    <div className="document-date">
                      {new Date(doc.upload_date).toLocaleDateString()}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="chat-container">
          <div className="chat-messages">
            {messages.length === 0 ? (
              <div className="welcome-message">
                <h2>Welcome to PDF Q&A Assistant!</h2>
                <p>Upload a PDF document and start asking questions about its content.</p>
                {!currentDocument && (
                  <p className="upload-prompt">👆 Click the "Upload PDF" button to get started</p>
                )}
                {currentDocument && (
                  <p className="ready-prompt">Ready to answer questions about: <strong>{currentDocument.filename}</strong></p>
                )}
              </div>
            ) : (
              messages.map((message, index) => (
                <div key={index} className={`message ${message.type}`}>
                  <div className="message-content">
                    {message.type === 'question' ? (
                      <div className="question-text">
                        <div className="message-header">
                          <strong>You:</strong>
                          {message.id && (
                            <button 
                              className="delete-message-btn"
                              onClick={() => handleDeleteMessage(message.id)}
                              disabled={deletingMessageId === message.id}
                              title="Delete this message"
                            >
                              {deletingMessageId === message.id ? '⏳' : '🗑️'}
                            </button>
                          )}
                        </div>
                        <div className="message-text">{message.content}</div>
                      </div>
                    ) : (
                      <div className="answer-text">
                        <div className="message-header">
                          <strong>Assistant:</strong>
                          {message.id && (
                            <button 
                              className="delete-message-btn"
                              onClick={() => handleDeleteMessage(message.id)}
                              disabled={deletingMessageId === message.id}
                              title="Delete this message"
                            >
                              {deletingMessageId === message.id ? '⏳' : '🗑️'}
                            </button>
                          )}
                        </div>
                        <div className="answer-content">{message.content}</div>
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
            {isLoading && (
              <div className="message answer">
                <div className="message-content">
                  <strong>Assistant:</strong>
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <form className="chat-input-form" onSubmit={handleSubmitQuestion}>
            <div className="input-container">
              <input
                type="text"
                value={currentQuestion}
                onChange={(e) => setCurrentQuestion(e.target.value)}
                placeholder={currentDocument ? `Ask a question about ${currentDocument.filename}...` : "Upload a PDF first to start asking questions..."}
                disabled={isLoading || !currentDocument}
                className="chat-input"
              />
              <button 
                type="submit" 
                disabled={isLoading || !currentQuestion.trim() || !currentDocument}
                className="send-btn"
              >
                {isLoading ? '...' : 'Send'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;