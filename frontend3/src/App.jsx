// import { useState } from 'react';
// import './App.css';

// function App() {
//   const [drawerOpen, setDrawerOpen] = useState(false);
//   const [drawerWidth, setDrawerWidth] = useState(600);
//   const targetPage = 2; // fixed page number

//   const handleOpenPdf = () => {
//     setDrawerOpen(true);
//   };

//   return (
//     <div style={{ display: 'flex', height: '100vh', width: '100vw', overflow: 'hidden' }}>
      
//       {/* Left Panel */}
//       <div style={{
//         flexGrow: 1,
//         backgroundColor: '#f4f6f8',
//         padding: '40px',
//         transition: 'width 0.3s ease'
//       }}>
//         <h2>🧠 Smart Document Chat</h2>

//         <button
//           onClick={handleOpenPdf}
//           style={{
//             padding: '10px 20px',
//             fontSize: '16px',
//             backgroundColor: '#007bff',
//             color: '#fff',
//             border: 'none',
//             borderRadius: '5px',
//             cursor: 'pointer',
//           }}
//         >
//           📂 Open Highlighted PDF
//         </button>
//       </div>

//       {/* Drag Divider */}
//       {drawerOpen && (
//         <div
//           style={{
//             width: '5px',
//             cursor: 'col-resize',
//             backgroundColor: '#ccc',
//             zIndex: 10
//           }}
//         />
//       )}

//       {/* PDF Viewer */}
//       {drawerOpen && (
//         <div style={{
//           width: `${drawerWidth}px`,
//           height: '100%',
//           display: 'flex',
//           flexDirection: 'column',
//           backgroundColor: '#fff',
//           boxShadow: '-2px 0 10px rgba(0,0,0,0.15)',
//           zIndex: 5
//         }}>
//           <div style={{
//             background: '#343a40',
//             color: '#fff',
//             padding: '10px 15px',
//             fontWeight: 'bold',
//             fontSize: '16px',
//             display: 'flex',
//             justifyContent: 'space-between',
//             alignItems: 'center'
//           }}>
//             📄 PDF Preview
//             <button
//               onClick={() => setDrawerOpen(false)}
//               style={{
//                 background: 'none',
//                 color: 'white',
//                 border: 'none',
//                 fontSize: '18px',
//                 cursor: 'pointer'
//               }}
//             >
//               ✕
//             </button>
//           </div>

//           <iframe
//             src={`/pdfjs/web/viewer.html?file=highlighted_output.pdf#page=${targetPage}`}
//             title="PDF Viewer"
//             style={{ width: '100%', height: '100%', border: 'none' }}
//           />
//         </div>
//       )}
//     </div>
//   );
// }

// export default App;

import { useState, useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './index.css';
import './App.css';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function App() {
  const [file, setFile] = useState(null);
  const [sentence, setSentence] = useState('');
  const [serverJson, setServerJson] = useState(null);
  const [loading, setLoading] = useState(false);

  // Drawer UI for PDF.js
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerWidth, setDrawerWidth] = useState(600);

  // Prefer the first actually highlighted page; fallback to first retrieved chunk; fallback to 1.
  const targetPage = useMemo(() => {
    const p0 = serverJson?.first_highlight_page;
    if (Number.isInteger(p0) && p0 > 0) return p0;
    const p1 = serverJson?.top_chunks?.[0]?.page;
    if (Number.isInteger(p1) && p1 > 0) return p1;
    return 1;
  }, [serverJson]);

  // cache-busting signature from backend so the iframe reloads updated PDF
  const cacheSig = serverJson?.hl_sig || '';

  const resetAll = () => {
    setFile(null);
    setSentence('');
    setServerJson(null);
    setDrawerOpen(false);
  };

  const onSubmit = async (e) => {
    e.preventDefault();
    if (!file || !sentence.trim()) return;
    setLoading(true);
    setServerJson(null);

    const form = new FormData();
    form.append('file', file);
    form.append('sentence', sentence);

    try {
      const res = await fetch(`${API}/analyze`, { method: 'POST', body: form });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.detail || data?.error || `HTTP ${res.status}`);
      setServerJson(data);
    } catch (err) {
      setServerJson({ ok: false, error: err?.message || 'network error' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', height: '100vh', width: '100vw', overflow: 'hidden' }}>
      
      {/* Left: uploader + results */}
      <div style={{ flex: 1, padding: '32px', background: '#f8f9fa', overflowY: 'auto' }}>
        <h2 style={{ marginTop: 0 }}>🧠 Smart Document Chat</h2>
        <p>Upload a PDF + a sentence. The backend retrieves, reasons, highlights, and answers with Gemini.</p>

        <form onSubmit={onSubmit} style={{ maxWidth: 720 }}>
          <div style={{ marginBottom: 16 }}>
            <label style={{ display: 'block', fontWeight: 600, marginBottom: 6 }}>PDF File</label>
            <input type="file" accept="application/pdf" onChange={e => setFile(e.target.files?.[0] || null)} />
          </div>

          <div style={{ marginBottom: 16 }}>
            <label style={{ display: 'block', fontWeight: 600, marginBottom: 6 }}>Sentence / Question</label>
            <input
              type="text"
              placeholder="Type any sentence..."
              value={sentence}
              onChange={e => setSentence(e.target.value)}
              style={{ width: '100%' }}
            />
          </div>

          <div style={{ display: 'flex', gap: 12 }}>
            <button type="submit" disabled={loading || !file || !sentence.trim()}>
              {loading ? 'Processing…' : '⬆️ Upload & Analyze'}
            </button>
            <button type="button" onClick={resetAll} style={{ background: '#212529', color: '#fff' }}>
              Reset
            </button>
            <button
              type="button"
              onClick={() => setDrawerOpen(true)}
              disabled={!serverJson}
              style={{ background: '#0d6efd', color: '#fff' }}
              title="Open the PDF viewer (loads highlighted_output.pdf)"
            >
              📂 Open Highlighted PDF
            </button>
          </div>
        </form>

        {/* Error card */}
        {serverJson && serverJson.ok === false && (
          <div className="card" style={{ marginTop: 16, maxWidth: 840, textAlign: 'left', borderColor: '#dc3545' }}>
            <h3 style={{ marginTop: 0, color: '#dc3545' }}>❌ Error</h3>
            <pre style={{ whiteSpace: 'pre-wrap' }}>{serverJson.error || 'Request failed'}</pre>
          </div>
        )}

        {/* Answer (Markdown-rendered, GPT-like styling) */}
        {serverJson?.final_answer && (
          <div className="card" style={{ marginTop: 24, maxWidth: 840, textAlign: 'left' }}>
            <div
              style={{
                fontSize: '1.125rem',     // ~18px
                lineHeight: 1.8,
                color: '#1b1f24'
              }}
              className="markdown-body"
            >
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  // Tweak common elements to feel “GPT-like”
                  p: ({ node, ...props }) => <p style={{ margin: '0 0 12px' }} {...props} />,
                  li: ({ node, ...props }) => <li style={{ margin: '6px 0' }} {...props} />,
                  strong: ({ node, ...props }) => <strong style={{ fontWeight: 700 }} {...props} />,
                  code: ({ inline, className, children, ...props }) => {
                    // Inline code look for formulas/symbols like v=u+at
                    return inline ? (
                      <code
                        style={{
                          background: '#f1f3f5',
                          padding: '0 4px',
                          borderRadius: 4,
                          fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
                          fontSize: '0.95em'
                        }}
                        {...props}
                      >
                        {children}
                      </code>
                    ) : (
                      <pre
                        style={{
                          background: '#0f172a',
                          color: '#e2e8f0',
                          padding: '14px',
                          borderRadius: 8,
                          overflowX: 'auto',
                          fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'
                        }}
                      >
                        <code {...props}>{children}</code>
                      </pre>
                    );
                  },
                  h1: (props) => <h1 style={{ fontSize: '1.35rem', margin: '0.75em 0 0.4em' }} {...props} />,
                  h2: (props) => <h2 style={{ fontSize: '1.25rem', margin: '0.75em 0 0.4em' }} {...props} />,
                  h3: (props) => <h3 style={{ fontSize: '1.15rem', margin: '0.75em 0 0.4em' }} {...props} />,
                  ul: (props) => <ul style={{ paddingLeft: '1.25em', margin: '0.5em 0' }} {...props} />,
                  ol: (props) => <ol style={{ paddingLeft: '1.25em', margin: '0.5em 0' }} {...props} />,
                  blockquote: (props) => (
                    <blockquote
                      style={{
                        borderLeft: '4px solid #e5e7eb',
                        margin: '0.75em 0',
                        padding: '0.25em 0 0.25em 1em',
                        color: '#334155'
                      }}
                      {...props}
                    />
                  ),
                  hr: (props) => <hr style={{ border: 0, borderTop: '1px solid #e9ecef', margin: '20px 0' }} {...props} />
                }}
              >
                {serverJson.final_answer}
              </ReactMarkdown>
            </div>
          </div>
        )}
      </div>

      {/* Right: PDF.js drawer viewer (same-origin relative file) */}
      {drawerOpen && (
        <div
          style={{
            width: `${drawerWidth}px`,
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            backgroundColor: '#fff',
            boxShadow: '-2px 0 10px rgba(0,0,0,0.15)',
            borderLeft: '1px solid #dee2e6',
            zIndex: 5,
          }}
        >
          <div
            style={{
              background: '#343a40',
              color: '#fff',
              padding: '10px 15px',
              fontWeight: 'bold',
              fontSize: '16px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            📄 PDF Preview
            <button
              onClick={() => setDrawerOpen(false)}
              style={{
                background: 'none',
                color: 'white',
                border: 'none',
                fontSize: '18px',
                cursor: 'pointer',
              }}
            >
              ✕
            </button>
          </div>

          <iframe
            // cache-bust with hl_sig so the viewer fetches the latest highlighted_output.pdf
            src={`/pdfjs/web/viewer.html?file=highlighted_output.pdf&v=${encodeURIComponent(cacheSig)}#page=${targetPage}`}
            title="PDF Viewer"
            style={{ width: '100%', height: '100%', border: 'none' }}
          />
        </div>
      )}
    </div>
  );
}
