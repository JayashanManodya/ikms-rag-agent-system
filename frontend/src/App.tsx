import React, { useState, useRef, useEffect } from 'react';
import { Upload, Send, FileText, Loader2, Bot, User, CheckCircle2, AlertCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import './App.css';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
const API_BASE_URL = import.meta.env.VITE_API_URL || (isLocal ? 'http://localhost:8000' : '/api');

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [isAnswering, setIsAnswering] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<{ type: 'success' | 'error', message: string } | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);

  const chatEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (file.type !== 'application/pdf') {
      setUploadStatus({ type: 'error', message: 'Only PDF files are supported.' });
      return;
    }

    setIsUploading(true);
    setUploadStatus(null);
    setFileName(file.name);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_BASE_URL}/index-pdf`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setUploadStatus({ type: 'success', message: `Successfully indexed ${response.data.chunks_indexed} chunks.` });
    } catch (error: any) {
      console.error('Upload failed:', error);
      const errorMsg = error.response?.data?.detail || error.message || 'Failed to upload and index PDF.';
      setUploadStatus({ type: 'error', message: errorMsg });
    } finally {
      setIsUploading(false);
    }
  };

  const handleSendMessage = async () => {
    if (!input.trim() || isAnswering) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsAnswering(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/qa`, { question: userMessage });
      setMessages(prev => [...prev, { role: 'assistant', content: response.data.answer }]);
    } catch (error) {
      console.error('QA request failed:', error);
      setMessages(prev => [...prev, { role: 'assistant', content: 'Sorry, I encountered an error while processing your request.' }]);
    } finally {
      setIsAnswering(false);
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar / Upload Panel */}
      <aside className="sidebar glass-panel">
        <div className="sidebar-header">
          <Bot size={28} className="icon-primary" />
          <h2>RAG Agent</h2>
        </div>

        <div className="upload-section">
          <h3>Document</h3>
          <p className="text-muted">Upload a PDF to start asking questions.</p>

          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileUpload}
            accept=".pdf"
            style={{ display: 'none' }}
          />

          <button
            className="upload-btn"
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
          >
            {isUploading ? <Loader2 className="animate-spin" /> : <Upload size={18} />}
            {isUploading ? 'Indexing...' : 'Upload PDF'}
          </button>

          {fileName && (
            <div className="file-info">
              <FileText size={16} />
              <span>{fileName}</span>
            </div>
          )}

          <AnimatePresence>
            {uploadStatus && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className={`status-msg ${uploadStatus.type}`}
              >
                {uploadStatus.type === 'success' ? <CheckCircle2 size={14} /> : <AlertCircle size={14} />}
                <span>{uploadStatus.message}</span>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="chat-area">
        <header className="chat-header glass-panel">
          <div className="status-indicator">
            <div className="pulse-dot"></div>
            <span>System Active</span>
          </div>
        </header>

        <div className="messages-container">
          {messages.length === 0 ? (
            <div className="welcome-screen">
              <Bot size={48} className="welcome-icon" />
              <h1>Ask anything about your PDF</h1>
              <p>Upload a document and I'll help you extract insights from it.</p>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, x: msg.role === 'user' ? 20 : -20 }}
                animate={{ opacity: 1, x: 0 }}
                className={`message-wrapper ${msg.role}`}
              >
                <div className="avatar">
                  {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
                </div>
                <div className="message-bubble glass-panel">
                  {msg.content}
                </div>
              </motion.div>
            ))
          )}
          {isAnswering && (
            <div className="message-wrapper assistant">
              <div className="avatar"><Bot size={16} /></div>
              <div className="message-bubble glass-panel loading">
                <div className="typing-dots">
                  <span></span><span></span><span></span>
                </div>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        <footer className="input-panel glass-panel">
          <div className="input-wrapper">
            <textarea
              placeholder="Ask a question..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
              rows={1}
            />
            <button
              className="send-btn"
              onClick={handleSendMessage}
              disabled={!input.trim() || isAnswering}
            >
              {isAnswering ? <Loader2 className="animate-spin" size={18} /> : <Send size={18} />}
            </button>
          </div>
        </footer>
      </main>
    </div>
  );
}

export default App;
