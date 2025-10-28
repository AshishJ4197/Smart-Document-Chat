# -------------------- file4.py (Cell 4 wrapped) --------------------
# coverage-aware selection + safe cleaning (balanced refinement)
import json, re
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

# ===== PASS 1: CONTEXT-AWARE SELECTION =====
class KeepIDs(BaseModel):
    ids: List[str]

# ===== PASS 2: CLEANING =====
BULLET_JOIN_RE = re.compile(r"(?:\s[•●▪‣■]\s|\sz\s)", re.I)
NUM_LIST_RE    = re.compile(r"\b(?:\(?[0-9]+|[ivxlcdm]+)\)?[.)]\s+\S+", re.I)
TAIL_RE = re.compile(
    r"(?i)\b(observe|write|group|classify|find out|according to|answer|project|exercise|assignment|"
    r"try this|can you|let us|discuss|for example|figure|fig\.|table|chart|science)\b"
)

def looks_joined(s: str) -> bool:
    return bool(BULLET_JOIN_RE.search(s)) or (len(re.findall(NUM_LIST_RE, s)) >= 2)

def trim_tail(s: str) -> str:
    m = TAIL_RE.search(s)
    return s[:m.start()].rstrip(" .,:;–—-") if m else s

def split_joined(s: str) -> List[str]:
    t = re.sub(r"\s+", " ", s).strip()
    t = re.sub(r"^(?:z)\s+", "• ", t)
    t = re.sub(r"\s(z)\s", " • ", t)
    parts = [p.strip(" •-–—") for p in t.split(" • ")]
    parts = [p for p in parts if len(p.split()) >= 2]
    return parts if len(parts) >= 2 else [s.strip()]

def strip_leading_markers(s: str) -> str:
    s = re.sub(r"^\s*[•●▪‣■]\s*", "", s)
    s = re.sub(r"^\s*(?:\(?[0-9]+|[ivxlcdm]+)\)?[.)-]\s*", "", s, flags=re.I)
    return s.strip()

def present_in_original(orig: str, piece: str) -> bool:
    if piece in orig:
        return True
    def norm(x: str):
        x = re.sub(r"\s*\(\s*", " (", x)
        x = re.sub(r"\s*\)\s*", ")", x)
        x = re.sub(r"\s+", " ", x)
        return x.strip()
    return norm(piece) in norm(orig)

class CleanItem(BaseModel):
    sid: str
    pieces: List[str]

class CleanBatch(BaseModel):
    items: List[CleanItem]

def _norm_tokens(s: str):
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return set(w for w in s.split() if len(w) > 2)

INTRO_PAT = re.compile(r"^(let us recall|remember that|you are already familiar)\b", re.I)
FURNITURE_PAT = re.compile(r"\b(science|figure|fig\.|table|chart|exercise|activity)\b", re.I)

def run_cell4(
    query: str,
    llm: Any,
    selected_ids_path: str = "highlight_ids.json",
    sentence_lookup_path: str = "sentence_lookup.json",
    clean_out_path: str = "ai_cleaned_sentences.json",
    answer_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    query : str
        The original user question.
    llm : Any
        The LangChain LLM used earlier (same instance type).
    selected_ids_path : str
        Path to Cell 3's highlight IDs JSON.
    sentence_lookup_path : str
        Path to Cell 3's sentence lookup JSON.
    clean_out_path : str
        Path to write cleaned sentences JSON.
    answer_text : Optional[str]
        Final answer text from Cell 2 (answer_response.content).

    Returns
    -------
    dict with keys:
        kept_count, pieces_count, clean_out_path, kept_ids, cleaned
    """
    base = Path(".")
    sel_ids_path = base / selected_ids_path
    lookup_path  = base / sentence_lookup_path
    clean_out    = base / clean_out_path

    # ---- load Cell 3 outputs
    with open(sel_ids_path, "r", encoding="utf-8") as f:
        selected_ids = json.load(f)

    with open(lookup_path, "r", encoding="utf-8") as f:
        sentence_lookup = json.load(f)

    candidates = []
    allowed = set(selected_ids)
    for sid in selected_ids:
        ent = sentence_lookup.get(sid)
        if not ent:
            continue
        candidates.append({
            "sid": sid,
            "page": int(ent["page"]),
            "sentence": ent["sentence"].strip()
        })

    # ===== PASS 1: CONTEXT-AWARE SELECTION =====
    class KeepIDs(BaseModel):
        ids: List[str]

    select_parser = PydanticOutputParser(pydantic_object=KeepIDs)
    select_instr = select_parser.get_format_instructions()

    def render_cands(cands):
        return "\n".join(f"- [sid: {c['sid']}] (page {c['page']}) {c['sentence']}" for c in cands)

    kept = []
    if candidates:
        select_prompt = ChatPromptTemplate.from_template("""
        You will choose the set of candidate sentences that best support the FINAL ANSWER.

        Inputs:
        - QUESTION: {query}
        - FINAL ANSWER: {answer}

        CANDIDATES (document order, with page numbers):
        {cands}

        Selection rules:
        - Keep ALL sentences that directly support or clarify the Final Answer.
        - INCLUDE useful introductory context if it frames the laws/definitions (e.g., “The following are the laws of…”).
        - Prefer the clearest, canonical formulations (e.g., numbered laws, definitions, rules).
        - Remove only:
          • Duplicates of the same claim
          • Obvious page furniture (headers, footers like “Science”, “Exercise”, “Figure…”, “Table…”)
          • Irrelevant reminders like “Let us recall…”, “Remember that…”
        - Do NOT remove sentences just to reduce the count.
        - Preserve order.

        Return ONLY this JSON:
        {format_instructions}
        """)

        try:
            keep_ids_chain = select_prompt | llm | select_parser
            keep_ids = keep_ids_chain.invoke({
                "query": query,
                "answer": (answer_text or ""),
                "cands": render_cands(candidates),
                "format_instructions": select_instr
            }).ids
        except Exception:
            keep_ids = []

        keep_set = set(i for i in keep_ids if i in allowed)
        kept = [c for c in candidates if c["sid"] in keep_set]

    # ===== Fallback if LLM fails =====
    if not kept:
        ans_toks = _norm_tokens(answer_text or "")
        scored = []
        for c in candidates:
            if INTRO_PAT.search(c["sentence"]) or FURNITURE_PAT.search(c["sentence"]):
                continue
            toks = _norm_tokens(c["sentence"])
            overlap = len(ans_toks & toks)
            scored.append((overlap, c["page"], c))
        scored.sort(key=lambda x: (-x[0], x[1]))
        kept = [t[2] for t in scored[:max(6, len(candidates)//2)]] or candidates[:2]

    # ===== PASS 2: CLEANING =====
    class CleanItem(BaseModel):
        sid: str
        pieces: List[str]

    class CleanBatch(BaseModel):
        items: List[CleanItem]

    clean_parser = PydanticOutputParser(pydantic_object=CleanBatch)
    clean_instr = clean_parser.get_format_instructions()

    clean_prompt = ChatPromptTemplate.from_template("""
    You will prepare the kept sentences for highlighting EXACTLY as they appear.

    For each entry:
    - If it is a normal prose sentence, return ONE piece (verbatim), only trimming an instruction tail if present.
    - If it is a run-on bullet line (joined by "•" / "z" / numbered items), split into separate pieces (order preserved).
    - Remove ONLY leading bullet/number markers from each piece.
    - Keep parentheses that belong to the phrase.
    - Do NOT rephrase or shorten normal sentences.
    - Ignore headings/labels or page furniture (e.g., "Science", "Exercise", "Figure", "Table").
    Return ONLY this JSON:
    {format_instructions}

    Entries:
    {entries}
    """)

    BATCH = 16
    kept_batches = [kept[i:i+BATCH] for i in range(0, len(kept), BATCH)]
    clean_results = []

    for group in kept_batches:
        entries_text = "\n".join(f"- [sid: {e['sid']}] {e['sentence']}" for e in group)
        chain = clean_prompt | llm | clean_parser
        out = chain.invoke({
            "entries": entries_text,
            "format_instructions": clean_instr
        })

        page_map = {e["sid"]: e["page"] for e in group}
        orig_map = {e["sid"]: e["sentence"] for e in group}

        for item in out.items:
            sid = item.sid
            orig = orig_map.get(sid, "")
            page = page_map.get(sid, 1)

            prose = not looks_joined(orig)

            if prose:
                cand = item.pieces[0] if item.pieces else orig
                cand = strip_leading_markers(trim_tail(cand)).strip() or strip_leading_markers(trim_tail(orig))
                if not present_in_original(orig, cand) or len(cand.split()) < max(3, int(0.4 * len(orig.split()))):
                    cand = strip_leading_markers(trim_tail(orig))
                clean_results.append({"sid": sid, "page": page, "pieces": [cand]})
            else:
                ai_pieces = item.pieces if item.pieces else split_joined(orig)
                pieces = []
                for p in ai_pieces:
                    p2 = strip_leading_markers(trim_tail(p)).strip()
                    if p2 and present_in_original(orig, p2):
                        pieces.append(p2)
                if not pieces:
                    pieces = [strip_leading_markers(trim_tail(orig))]
                clean_results.append({"sid": sid, "page": page, "pieces": pieces})

    # ---- write
    with open(clean_out, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=2, ensure_ascii=False)

    return {
        "kept_count": len(kept),
        "pieces_count": sum(len(r["pieces"]) for r in clean_results),
        "clean_out_path": str(clean_out),
        "kept_ids": [c["sid"] for c in kept],
        "cleaned": clean_results,
    }

# ... keep all your existing code above ...

def prepare_cleaned_sentences(*, llm, query: str, final_answer):
    """
    Backend expects this name. Calls run_cell4 and returns its result.
    Reads:  highlight_ids.json, sentence_lookup.json
    Writes: ai_cleaned_sentences.json
    """
    answer_text = final_answer if isinstance(final_answer, str) else str(final_answer)
    return run_cell4(
        query=query,
        llm=llm,
        selected_ids_path="highlight_ids.json",
        sentence_lookup_path="sentence_lookup.json",
        clean_out_path="ai_cleaned_sentences.json",
        answer_text=answer_text,
    )

__all__ = ["prepare_cleaned_sentences", "run_cell4"]
