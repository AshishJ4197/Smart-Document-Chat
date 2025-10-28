const express = require('express');
const path = require('path');

const app = express();
const PORT = 3001;

// Serve built React files
app.use(express.static(path.join(__dirname, 'frontend2', 'build')));

// Serve PDF.js viewer and assets
app.use('/pdfjs', express.static(path.join(__dirname, 'frontend2', 'public', 'pdfjs')));

// Catch-all route to serve index.html (React routing)
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'frontend2', 'build', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`✅ Server running at http://localhost:${PORT}`);
});
