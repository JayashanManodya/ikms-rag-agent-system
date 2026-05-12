import React, { useState, useRef, useEffect } from 'react';
import { Upload, Send, FileText, Loader2, Bot, User, CheckCircle2, AlertCircle, AlertTriangle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import './App.css';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

const rawApiUrl = import.meta.env.VITE_API_URL;
if (!rawApiUrl) {
  throw new Error(
    'VITE_API_URL is not set. Add it to frontend/.env.local or your host (e.g. Vercel) environment variables.'
  );
}
const API_BASE_URL = rawApiUrl.replace(/\/$/, '');

/** Yellow sidebar note: free-tier testing, larger PDFs. */
const DEMO_FREE_HOST_TITLE = 'Running on free hosting (testing)';
const DEMO_FREE_HOST_BODY =
  'This demo uses free servers, so they may sleep when nobody’s using the app. Larger PDFs also need more time to upload and process. Delays are expected—not a bug. Thanks for sticking with us while things spin up.';

function FreeHostingNotice({ role = 'status' }: { role?: 'status' | 'note' }) {
  return (
    <div className="demo-notice" role={role}>
      <AlertTriangle size={18} className="demo-notice-icon" aria-hidden />
      <div className="demo-notice-copy">
        <strong className="demo-notice-title">{DEMO_FREE_HOST_TITLE}</strong>
        <span>{DEMO_FREE_HOST_BODY}</span>
      </div>
    </div>
  );
}

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
      setUploadStatus({ type: 'error', message: 'Please choose a PDF file (.pdf).' });
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
      const n = response.data.chunks ?? response.data.chunks_indexed;
      setUploadStatus({
        type: 'success',
        message:
          typeof n === 'number'
            ? `All set! Your PDF is ready—we’ve read ${n} section${n === 1 ? '' : 's'} from it. Ask a question on the right.`
            : (response.data.message as string) || 'Your PDF is ready. You can ask questions about it now.',
      });
    } catch (error: any) {
      console.error('Upload failed:', error);
      const errorMsg = error.response?.data?.detail || error.message || 'We couldn’t finish reading your PDF. Check your connection and try again.';
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
      setMessages(prev => [...prev, { role: 'assistant', content: 'Something went wrong answering that—please try again in a moment.' }]);
    } finally {
      setIsAnswering(false);
    }
  };

  return (
    <div className="app-shell">
      <div className="app-container">
      {/* Sidebar / Upload Panel */}
      <aside className="sidebar glass-panel">
        <div className="sidebar-header">
          <Bot size={28} className="icon-primary" />
          <h2>Ask your PDF</h2>
        </div>

        <div className="upload-section">
          <h3>Your document</h3>
          <p className="text-muted">Upload a PDF, then chat with an assistant that answers using only what’s in your file.</p>

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
            {isUploading ? 'Reading your file…' : 'Upload your PDF'}
          </button>

          <AnimatePresence>
            {isUploading && (
              <motion.div
                key="upload-free-hosting-hint"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
              >
                <FreeHostingNotice role="status" />
              </motion.div>
            )}
          </AnimatePresence>

          {fileName && (
            <div className="file-info">
              <FileText size={16} />
              <span>{fileName}</span>
            </div>
          )}

          <AnimatePresence>
            {uploadStatus && (
              <motion.div
                key="upload-status-block"
                className="upload-status-stack"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
              >
                <div className={`status-msg ${uploadStatus.type}`}>
                  {uploadStatus.type === 'success' ? <CheckCircle2 size={14} /> : <AlertCircle size={14} />}
                  <span>{uploadStatus.message}</span>
                </div>
                {uploadStatus.type === 'success' && <FreeHostingNotice role="note" />}
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
            <span>Ready to help</span>
          </div>
        </header>

        <div className="messages-container">
          {messages.length === 0 ? (
            <div className="welcome-screen">
              <Bot size={48} className="welcome-icon" />
              <h1>Chat with your document</h1>
              <p>First upload a PDF on the left. Then ask plain-language questions. we answer using the text in your file, not random web results.</p>
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
              placeholder="Type your question about the PDF…"
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

      <footer className="site-footer">
        <div className="site-footer-inner">
          <p className="site-footer-copy">
            Built by{' '}
            <a
              className="site-footer-link"
              href="https://www.jayashan.online/"
              target="_blank"
              rel="noopener noreferrer"
            >
              Jayashan Manodya
            </a>
            <span className="site-footer-sep" aria-hidden>
              {' '}
              ·{' '}
            </span>
            © {new Date().getFullYear()}. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
