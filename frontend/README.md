# PDF Q&A Assistant

A React-based web application that allows users to upload PDF documents and ask questions about their content using AI-powered document analysis.

## Features

- PDF Upload: Upload and process PDF documents
- Document Management: View and manage multiple uploaded documents
- Interactive Q&A: Ask questions about uploaded PDF content and receive AI-generated answers
- Chat History: Persistent chat history for each document
- Message Management: Delete individual messages or clear entire chat history
- Document Deletion: Remove documents and associated chat history
- Real-time Status: Upload progress and status notifications

## Prerequisites

- Node.js (v14 or higher)
- npm or yarn
- Backend API server running on port 8000 (or configured URL)

## Installation

1. Install dependencies:
```bash
npm install
```

2. Set up environment variables:
Create a `.env` file in the root directory:
```env
REACT_APP_API_URL=http://localhost:8000
```

3. Start the development server:
```bash
npm start
```

The application will open in your browser at `http://localhost:3000`.


## Styling

The application uses custom CSS with a modern, clean design featuring:
- Responsive layout with sidebar and main content area
- Dark theme with blue accent colors
- Smooth animations and hover effects
- Loading indicators and status messages
- Mobile-friendly design

## Error Handling

The application includes comprehensive error handling for:
- Network errors during API calls
- File upload failures
- Invalid file types (non-PDF files)
- Missing documents or chat history
- Deletion confirmations to prevent accidental data loss

## Development

### Available Scripts

- `npm start` - Run development server
- `npm build` - Build for production
- `npm test` - Run test suite
- `npm eject` - Eject from Create React App

### Environment Variables

- `REACT_APP_API_URL` - Backend API base URL (default: http://0.0.0.0:8000)

