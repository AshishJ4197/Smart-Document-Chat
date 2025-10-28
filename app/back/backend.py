# -------------------- backend.py (Cell 1 + Cell 2, using uploaded filename as source) --------------------
# 0) Windows OpenMP clash fix (FAISS + spaCy/numpy). Must be FIRST.
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from uuid import uuid4
from pathlib import Path
from typing import List, Optional
from contextlib import contextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# LangChain + Google GenAI
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import SpacyTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

import fitz  # PyMuPDF
import json
import traceback
import shutil

# === NEW: call Cells 3/4/5 modules ===
# (These are the files you asked for earlier: file3.py, file4.py, file5.py)
# They expose tiny functions we can call from the backend.
try:
    from file3 import build_support_ids  # (retrieved, query, final_answer, llm) -> {"ids":[...]}
    from file4 import prepare_cleaned_sentences  # (llm, query, final_answer) -> writes ai_cleaned_sentences.json
    from file5 import highlight_pdf  # (pdf_in: Path, pdf_out: Path) -> {"total_spans": int, "first_highlight_page": Optional[int]}
except Exception:
    # If the user hasn’t dropped these files in yet, we’ll still return the answer without highlights.
    build_support_ids = None
    prepare_cleaned_sentences = None
    highlight_pdf = None

# ---------- config ----------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
VECTOR_DIR = BASE_DIR / "vector_index"
OUTPUT_TXT = BASE_DIR / "cell2_output.txt"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

# Where the web viewer expects the highlighted file:
FRONTEND_PDFJS_DIR = BASE_DIR.parent.parent / "frontend3" / "public" / "pdfjs" / "web"
VIEWER_HIGHLIGHTED = FRONTEND_PDFJS_DIR / "highlighted_output.pdf"

@contextmanager
def pushd(new_dir: Path):
    """Temporarily set the working directory so file3/file4/file5 write JSONs next to backend.py."""
    prev = Path.cwd()
    os.chdir(str(new_dir))
    try:
        yield
    finally:
        os.chdir(str(prev))

# ---------- Google GenAI (enter API key here) ----------
GOOGLE_API_KEY = ""  # <- replace with your key;
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    api_key=GOOGLE_API_KEY,
)
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY,
)

# ---------- FastAPI ----------
app = FastAPI(title="Smart Doc Chat Backend (Cells 1+2)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/files", StaticFiles(directory=UPLOAD_DIR), name="files")

@app.get("/")
def root():
    return {"ok": True, "msg": "Backend (Cells 1+2) alive"}

@app.post("/analyze")
async def analyze(
    file: Optional[UploadFile] = File(None),
    pdf: Optional[UploadFile] = File(None),
    sentence: str = Form(...),
):
    """
    Exact Cell 1 + Cell 2 logic, with 'source' fields set to the uploaded filename.
    - SpacyTextSplitter(pipeline="en_core_web_sm", chunk_size=1500, chunk_overlap=375)
    - FAISS index_name='class_10_phy'
    - retriever similarity k=5
    - Writes a single consolidated cell2_output.txt (reasoning + answer + supporting chunks)
    - NEW: Runs Cells 3→4→5 and writes highlighted PDF to frontend:
        ai_bot/frontend3/public/pdfjs/web/highlighted_output.pdf
    """
    try:
        upload = file or pdf
        if upload is None:
            raise HTTPException(status_code=422, detail="No file provided. Use 'file' or 'pdf'.")

        # Save upload to disk
        original_name = (upload.filename or "upload.pdf")
        ext = original_name.split(".")[-1].lower()
        if ext != "pdf":
            ext = "pdf"
        stored_name = f"{uuid4().hex}.{ext}"
        stored_path = UPLOAD_DIR / stored_name
        stored_path.write_bytes(await upload.read())

        # --------------------------
        # Cell 1 — Load, split, index
        # --------------------------
        with fitz.open(stored_path) as d:
            num_pages = d.page_count
        docs = PyMuPDFLoader(str(stored_path)).load()

        text_splitter = SpacyTextSplitter(pipeline="en_core_web_sm", chunk_size=2000, chunk_overlap=500)
        split_docs = text_splitter.split_documents(docs)

        chunk_metadata = []
        uuid_lookup = {}
        processed_docs: List[Document] = []

        # Use the uploaded filename as the 'source' everywhere
        user_source = Path(original_name).name

        for doc in split_docs:
            uid = uuid4().hex
            page = int(doc.metadata.get("page", 0)) + 1

            metadata = {**doc.metadata, "uuid": uid, "page": page, "source": user_source}
            processed_docs.append(Document(page_content=doc.page_content, metadata=metadata))

            entry = {
                "uuid": uid,
                "page": page,
                "text": doc.page_content,
                "source": user_source,
            }
            chunk_metadata.append(entry)
            uuid_lookup[uid] = entry

        # Save chunk metadata (same filenames as in your cells)
        (BASE_DIR / "sentence_metadata.json").write_text(
            json.dumps(chunk_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (BASE_DIR / "uuid_lookup.json").write_text(
            json.dumps(uuid_lookup, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Create and reload FAISS vector store (index_name matches cell 1)
        index_name = "class_10_phy"
        vector_store = FAISS.from_documents(processed_docs, embedding=embedding_model)
        vector_store.save_local(folder_path=str(VECTOR_DIR), index_name=index_name)

        vector_store = FAISS.load_local(
            folder_path=str(VECTOR_DIR),
            index_name=index_name,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True,
        )
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        # --------------------------
        # Cell 2 — Reasoning + Answer
        # --------------------------
        query = (sentence or "").strip() or "What are the laws of reflection?"
        retrieved = retriever.invoke(query)
        relevant = [doc.page_content for doc in retrieved]

        reasoning_prompt = ChatPromptTemplate.from_template("""
You are analyzing textbook excerpts to answer a question.

📘 Question:
{query}

📚 Textbook Chunks:
{relevant_docs}

Instructions:
1) Extract ALL key facts that directly address the question.
   - Always include the main laws, rules, definitions, named results, and formulas.
   - Also include any interpretation conventions needed to read/apply those items
     (generic rule: if a sign/value/symbol implies a meaning or case, capture that mapping).
   - Include any applicability constraints the source states (domains, ranges, bounds, validity conditions).
   - Include any concise conceptual statement that explains what an item represents or relates
     (e.g., “gives the relationship between …”, “is expressed as the ratio of …”).
   - Keep numbers, symbols, equations, variable names, constants, and technical terms EXACTLY as written.
   - Ignore only text that is clearly unrelated.

2) Create a brief GLOSSARY for items used in those facts:
   - List abbreviations, acronyms, symbols, variables, constants, or units that appear in the facts/equations.
   - For each, give its meaning exactly as stated in the text (precise and minimal).
   - If the text maps a sign/value to a meaning (e.g., “positive means …”), record that mapping once.

3) Identify the list of ENTITIES that directly answer the question.
   - Each entity should be one atomic item (law, rule, definition, condition, or equation).

4) Provide a short JUSTIFICATION based only on the extracted facts.

Return EXACTLY in this format:

FACTS:
- <fact1>
- <fact2>
...

GLOSSARY:
- <token> = <meaning>
- <token> = <meaning>
...

ENTITIES:
- <entity1>
- <entity2>
...

JUSTIFICATION: <one short sentence based only on the facts>
""")
        reasoning_chain = reasoning_prompt | llm
        reasoning_response = reasoning_chain.invoke({
            "query": query,
            "relevant_docs": relevant
        })
        reasoning_text = getattr(reasoning_response, "content", str(reasoning_response))

        answer_prompt = ChatPromptTemplate.from_template("""
You are a tutor writing a clear, textbook-style answer.

📘 Question:
{query}

🔎 Reasoning Output:
{reasoning}

Goal:
Return a polished answer that includes the core item(s) (e.g., formulas/definitions) PLUS the minimal context
needed to understand/apply them. The style should fit the question.

Formatting policy (choose ONE style):
- If the question effectively asks for ONE core item (e.g., a single definition or formula),
  write a compact PARAGRAPH (1-3 sentences). Include the equation inline and any essential
  interpretation conventions or constraints in the same paragraph.
- If there are 2-5 core items, use a clean NUMBERED LIST (1 level only; no sub-bullets unless unavoidable).
- If the question names “laws/rules/steps” or there are >5 items, use a numbered list.

Content rules (apply regardless of style):
- For EACH core item:
  • Include its formal statement/equation verbatim (symbols/notation unchanged).
  • Add one brief generic interpretation (e.g., “gives the relationship between …”, “ratio of …”).
  • Attach any essential interpretation conventions (e.g., sign/value → meaning) and applicability constraints.
  • Inline the minimal meanings of symbols/variables/constants needed to read the item on its own
    (use the provided glossary/facts; keep it brief).
- DO NOT move essentials to “Additional Points.” Essentials must appear with the item.
- Preserve all numbers, symbols, and precise terms from the reasoning facts/glossary.
- Prefer exact phrasing from the excerpts for formal names and equations.
- Avoid verbose sub-lists. One level only. Keep it crisp and readable.

Additional Points (optional):
- Add a section titled exactly: **Additional Points:**
- Include 0–2 short bullets that are helpful but NOT required to interpret/apply the main items
  (e.g., a common pitfall, a tiny tip, or a closely related consequence).
- Do NOT duplicate content already stated in the main section.

Special handling:
- If the question looks like multiple-choice, clearly state the correct option(s) with a brief justification.
- If a length is requested (e.g., “in N words”), match it (±10%) prioritizing facts from the excerpts).
- If explicitly asked to compare/contrast, present a 2-column Markdown table.

Return ONLY the final answer text (main section + the **Additional Points** section, if any).
""")
        answer_chain = answer_prompt | llm
        answer_response = answer_chain.invoke({
            "query": query,
            "reasoning": reasoning_text
        })
        final_answer_text = getattr(answer_response, "content", str(answer_response))

        # Supporting chunks (Cell 2 print-equivalent)
        supporting_blocks = []
        for d in retrieved:
            uid = d.metadata.get("uuid")
            entry = uuid_lookup.get(uid)
            if entry:
                supporting_blocks.append(
                    f"🆔 {uid} | 📄 Page: {entry['page']}\n{entry['text']}\n" + "-" * 60
                )

        # Write ONE consolidated text file
        out_lines = []
        out_lines.append("🔍 Reasoning Phase:\n" + reasoning_text)
        out_lines.append("=" * 80)
        out_lines.append("✅ Final Answer:\n" + final_answer_text)
        out_lines.append("=" * 80)
        out_lines.append("\n🔍 Supporting Chunks:\n" + "-" * 60)
        out_lines.extend(supporting_blocks)
        OUTPUT_TXT.write_text("\n".join(out_lines), encoding="utf-8")

        # Also return JSON for the UI
        top_chunks = []
        for d in retrieved:
            uid = d.metadata.get("uuid")
            page = int(d.metadata.get("page", 0))
            text = d.page_content or ""
            preview = (text[:300] + "…") if len(text) > 300 else text
            top_chunks.append({"uuid": uid, "page": page, "preview": preview})

        # --------------------------
        # NEW: Cells 3 → 4 → 5, then place the highlighted PDF where the viewer expects it
        # --------------------------
        highlighted_url = None
        highlighter_meta = None
        first_highlight_page = None
        hl_sig = None

        try:
            if build_support_ids and prepare_cleaned_sentences and highlight_pdf:
                # Ensure all JSON I/O happens beside backend.py
                with pushd(BASE_DIR):
                    # Cell 3
                    build_support_ids(
                        retrieved=retrieved,
                        query=query,
                        final_answer=final_answer_text,
                        llm=llm
                    )
                    # Cell 4
                    prepare_cleaned_sentences(
                        llm=llm,
                        query=query,
                        final_answer=final_answer_text
                    )

                    # >>> NEW: infer first page from ai_cleaned_sentences.json (Cell 4 output)
                    try:
                        cleaned_path = BASE_DIR / "ai_cleaned_sentences.json"
                        if cleaned_path.exists():
                            cleaned = json.loads(cleaned_path.read_text(encoding="utf-8"))
                            pages = [
                                int(row["page"])
                                for row in cleaned
                                if int(row.get("page", 0)) > 0 and any(p.strip() for p in row.get("pieces", []))
                            ]
                            if pages:
                                first_highlight_page = min(pages)
                    except Exception:
                        # keep None if anything goes wrong; UI will fallback
                        pass

                    # Cell 5 → write to a temp output first (same folder as backend), then move
                    backend_highlight_tmp = BASE_DIR / "highlighted_output.pdf"
                    res = highlight_pdf(pdf_in=stored_path, pdf_out=backend_highlight_tmp)
                    # keep whatever metadata Cell 5 returns too
                    highlighter_meta = res

                # Ensure viewer location exists and copy the file
                FRONTEND_PDFJS_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(backend_highlight_tmp, VIEWER_HIGHLIGHTED)

                # URL the frontend iframe already uses
                highlighted_url = "/pdfjs/web/highlighted_output.pdf"

                # cache-busting signature for the frontend iframe
                hl_sig = uuid4().hex

        except Exception as e:
            # We keep the rest of the response even if highlighting fails
            traceback.print_exc()
            highlighted_url = None
            highlighter_meta = {"error": str(e)}

        return {
            "ok": True,
            "received_sentence": sentence,
            "original_filename": original_name,
            "stored_filename": stored_name,
            "file_url": f"/files/{stored_name}",
            "num_pages": num_pages,
            "final_answer": final_answer_text,
            "reasoning": reasoning_text,
            "top_chunks": top_chunks,
            "output_text_path": str(OUTPUT_TXT),
            # NEW: highlighted pdf info for your UI
            "highlighted_pdf_url": highlighted_url,   # -> "/pdfjs/web/highlighted_output.pdf"
            "highlighter_meta": highlighter_meta,
            "first_highlight_page": first_highlight_page,  # -> used by UI to open the first page with highlights
            "hl_sig": hl_sig,  # cache-busting token for the viewer iframe
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})

# Dev helper: uvicorn backend:app --reload --port 8000
# ---------------------------------------------------------------------------
