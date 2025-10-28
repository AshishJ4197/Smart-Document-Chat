const pdfInput = document.getElementById('pdfInput');
const openViewer = document.getElementById('openViewer');
const sidePanel = document.getElementById('sidePanel');
const closePanel = document.getElementById('closePanel');
const pdfCanvas = document.getElementById('pdfCanvas');

let pdfData = null;

pdfInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file && file.type === 'application/pdf') {
    const reader = new FileReader();
    reader.onload = function () {
      pdfData = new Uint8Array(this.result);
      openViewer.disabled = false;
    };
    reader.readAsArrayBuffer(file);
  } else {
    alert('Please select a valid PDF file.');
  }
});

openViewer.addEventListener('click', () => {
  sidePanel.classList.add('active');
  if (pdfData) {
    renderPDF(pdfData);
  }
});

closePanel.addEventListener('click', () => {
  sidePanel.classList.remove('active');
});

function renderPDF(data) {
  const loadingTask = pdfjsLib.getDocument({ data });
  loadingTask.promise.then((pdf) => {
    // Load the first page for now
    pdf.getPage(1).then((page) => {
      const viewport = page.getViewport({ scale: 1.5 });
      const context = pdfCanvas.getContext('2d');
      pdfCanvas.height = viewport.height;
      pdfCanvas.width = viewport.width;

      const renderContext = {
        canvasContext: context,
        viewport: viewport
      };
      page.render(renderContext);
    });
  }).catch(err => {
    console.error('Error loading PDF:', err);
  });
}
