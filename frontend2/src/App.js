import React, { useState, useRef } from 'react';
import PDFViewer from './PDFviewer';

function App() {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerWidth, setDrawerWidth] = useState(600);
  const [targetPage, setTargetPage] = useState(1);
  const isDragging = useRef(false);

  const startResize = () => {
    isDragging.current = true;
    document.addEventListener('mousemove', handleResize);
    document.addEventListener('mouseup', stopResize);
  };

  const handleResize = (e) => {
    if (!isDragging.current) return;
    const newWidth = window.innerWidth - e.clientX;
    if (newWidth > 300 && newWidth < window.innerWidth - 200) {
      setDrawerWidth(newWidth);
    }
  };

  const stopResize = () => {
    isDragging.current = false;
    document.removeEventListener('mousemove', handleResize);
    document.removeEventListener('mouseup', stopResize);
  };

  const handleOpenPdf = async () => {
    try {
      const res = await fetch('/chunks_with_positions.json');
      const data = await res.json();
      setTargetPage(data?.[0]?.page || 1);
    } catch (err) {
      console.error('Error loading metadata:', err);
      setTargetPage(1);
    } finally {
      setDrawerOpen(true);
    }
  };

  return (
    <div style={{ display: 'flex', height: '100vh', width: '100vw', overflow: 'hidden' }}>
      
      {/* 🧠 Main Content */}
      <div style={{
        flexGrow: 1,
        backgroundColor: '#f4f6f8',
        padding: '40px',
        transition: 'width 0.3s ease'
      }}>
        <h2>🧠 Smart Document Chat</h2>
        <button
          onClick={handleOpenPdf}
          style={{
            padding: '10px 20px',
            fontSize: '16px',
            backgroundColor: '#007bff',
            color: '#fff',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
          }}
        >
          📂 Open Highlighted PDF
        </button>
      </div>

      {/* 📏 Draggable Divider */}
      {drawerOpen && (
        <div
          onMouseDown={startResize}
          style={{
            width: '5px',
            cursor: 'col-resize',
            backgroundColor: '#ccc',
            zIndex: 10
          }}
        />
      )}

      {/* 📄 PDF Viewer Panel */}
      {drawerOpen && (
        <div style={{
          width: `${drawerWidth}px`,
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          backgroundColor: '#fff',
          boxShadow: '-2px 0 10px rgba(0,0,0,0.15)',
          zIndex: 5
        }}>
          {/* Header */}
          <div style={{
            background: '#343a40',
            color: '#fff',
            padding: '10px 15px',
            fontWeight: 'bold',
            fontSize: '16px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            📄 PDF Preview
            <button
              onClick={() => setDrawerOpen(false)}
              style={{
                background: 'none',
                color: 'white',
                border: 'none',
                fontSize: '18px',
                cursor: 'pointer'
              }}
            >
              ✕
            </button>
          </div>

          {/* Body */}
          <div style={{ flex: 1, overflow: 'auto' }}>
            <PDFViewer file="/highlighted_output.pdf" page={targetPage} />
          </div>
        </div>
      )}
    </div>
  );
}

export default App;

