from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
import os

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Backend is running."}

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    os.makedirs("temp_pdfs", exist_ok=True)
    file_path = f"temp_pdfs/{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Extract text from PDF
    try:
        reader = PdfReader(file_path)
        all_pages = {}
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            all_pages[f"page_{i+1}"] = text.strip() if text else ""

        return JSONResponse(content={
            "filename": file.filename,
            "pages": all_pages
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
