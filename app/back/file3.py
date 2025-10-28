# ===== file3.py — Cell 3 (v2 repair/merge + v1 minimal LLM selection) =====
import json, re
from math import ceil
from typing import List, Dict, Any

import syntok.segmenter as segmenter
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel

# --------- repair/merge logic (identical to your cell, plus minimal enum-carry tweak) ---------
OP_CHARS = r"<>=±+\-*/^:≈≠≤≥→∝"
OP_TAIL_RE    = re.compile(rf"[{re.escape(OP_CHARS)}]\s*$")           # e.g., "0 <"
OP_HEAD_RE    = re.compile(rf"^\s*[{re.escape(OP_CHARS)}]")           # e.g., "< i"
CLOSE_HEAD_RE = re.compile(r"^\s*[\)\]\}.,;:]")                        # e.g., ") then"
LOWVAR_HEAD   = re.compile(r"^\s*[a-z]\b", re.I)                       # e.g., "i", "r"
ENUM_HEAD_RE  = re.compile(r"^\s*(\(?[ivxlcdmIVXLCDM]+\)|\(?[a-zA-Z]\)|\d+[.)])\s+")  # (i), 1., a)

# NEW: tiny detectors (same as notebook cell)
PAREN_ONLY      = re.compile(r"^\(\s*[^()]{1,12}\s*\)\.?\s*$")         # e.g., "( f )", "(h′)", "(9.8)"
ENDS_WITH_COLON = re.compile(r":\s*$")
# NEW (minimal): bare enumeration token like "(ii)", "(b)", "(2)" -> to be carried to next sentence
ENUM_PAREN_ONLY = re.compile(r"^\(\s*(?:[ivxlcdmIVXLCDM]+|[a-zA-Z]|\d+)\s*\)\.?\s*$")

def _norm_spaces(s: str) -> str:
    import re as _re
    return _re.sub(r"\s+", " ", (s or "").strip())

def unbalanced_paren_or_quote(s: str) -> bool:
    return (s.count("(") != s.count(")")) or (s.count('"') % 2 != 0) or (s.count("'") % 2 != 0)

def is_parenthetical_fragment(s: str) -> bool:
    """Tiny stand-alone parenthetical like '( f )', '(h′)', '(9.8)'. Keeps variables/equation refs with prior line."""
    return bool(PAREN_ONLY.match(_norm_spaces(s)))

def should_merge(prev: str, curr: str) -> bool:
    """Merge only when the break is clearly structural (paren/math), not normal prose."""
    import re as _re
    if not prev or not curr:
        return False
    # --- existing rules ---
    if ENUM_HEAD_RE.search(curr) and not (unbalanced_paren_or_quote(prev) or OP_TAIL_RE.search(prev)):
        return False
    if unbalanced_paren_or_quote(prev):
        return True
    if OP_TAIL_RE.search(prev) and (OP_HEAD_RE.search(curr) or CLOSE_HEAD_RE.search(curr) or LOWVAR_HEAD.search(curr)):
        return True
    if CLOSE_HEAD_RE.search(curr) and not _re.search(r"[.?!:]$", prev.strip()):
        return True
    # --- NEW: stitch tiny parenthetical fragment into the previous sentence unless the previous ends with a colon
    if is_parenthetical_fragment(curr) and not ENDS_WITH_COLON.search(prev.strip()):
        return True
    return False

def repair_fragments(sentences: List[str]) -> List[str]:
    """Single L→R pass; minimal touch. Never deletes text.
    Minimal tweak: if a standalone '(ii)'/'(b)'/'(2)' line is seen, carry it forward and prefix the next sentence.
    """
    out: List[str] = []
    pending_enum: str = ""  # holds a bare enumeration token to attach to the next sentence

    for s in sentences:
        s = _norm_spaces(s)
        if not s:
            continue

        # If this line is ONLY an enumeration token like "( ii )", don't merge to previous; stash for the next sentence.
        if ENUM_PAREN_ONLY.match(s):
            pending_enum = s
            continue

        # If we have a pending enumeration token, prepend it to the current sentence (the "next" sentence).
        if pending_enum:
            s = f"{pending_enum} {s}"
            pending_enum = ""

        if out and should_merge(out[-1], s):
            out[-1] = _norm_spaces(out[-1] + " " + s)
        else:
            out.append(s)

    # If the very last item was a dangling enumeration token (no next sentence existed), keep it as its own line.
    if pending_enum:
        out.append(pending_enum)

    # if something still unbalanced, greedily attach next until balanced
    i, fixed = 0, []
    while i < len(out):
        cur = out[i]
        if unbalanced_paren_or_quote(cur) and i + 1 < len(out):
            combo = cur + " " + out[i+1]
            i += 2
            while i < len(out) and unbalanced_paren_or_quote(combo):
                combo += " " + out[i]
                i += 1
            fixed.append(_norm_spaces(combo))
        else:
            fixed.append(cur)
            i += 1
    return fixed

# --------- public API ---------
def run_file3(*, retrieved: List[Any], query: str, final_answer_text: str, llm) -> Dict[str, Any]:
    """
    Execute Cell 3 on top of provided `retrieved` chunks, using the same LLM.
    Side-effects:
      - writes `sentence_lookup.json`
      - writes `highlight_ids.json`
    Returns:
      {"supporting_sentence_ids": [...]}  # in document order
    """
    # --------- segment retrieved into repaired candidates ---------
    sentence_candidates = []   # [{sid, sentence}]
    sentence_lookup = {}       # sid -> full entry
    sid_counter = 1

    for doc in retrieved:
        chunk_uuid = doc.metadata["uuid"]
        page = doc.metadata["page"]
        source = doc.metadata["source"]
        chunk_text = doc.page_content or ""

        raw_sents: List[str] = []
        paragraphs = segmenter.analyze(chunk_text)
        for paragraph in paragraphs:
            for sentence in paragraph:
                sent_text = " ".join(token.value for token in sentence).strip()
                if sent_text:
                    raw_sents.append(sent_text)

        fixed_sents = repair_fragments(raw_sents)

        for sentence_text in fixed_sents:
            sid = f"s{sid_counter}"
            sid_counter += 1
            entry = {
                "sid": sid,
                "sentence": sentence_text,
                "page": page,
                "chunk_uuid": chunk_uuid,
                "source": source
            }
            sentence_candidates.append({"sid": sid, "sentence": sentence_text})
            sentence_lookup[sid] = entry

    # 🔒 Save sentence metadata
    with open("sentence_lookup.json", "w", encoding="utf-8") as f:
        json.dump(sentence_lookup, f, indent=2, ensure_ascii=False)

    # --------- v1-style minimal LLM selection (unchanged logic) ---------
    class SupportIDs(BaseModel):
        ids: List[str]

    parser = PydanticOutputParser(pydantic_object=SupportIDs)
    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_template("""
    You are helping a student understand which textbook sentences support a given answer.

    You are given:
    - A user question
    - An answer to that question
    - A list of textbook sentences, each with a short ID

    Your task is to identify the **minimum set** of sentence IDs that directly support the answer.

    Only include:
    - Sentences that clearly explain the final answer
    - Definitions, laws, rules, or steps that are explicitly used in the answer
    - The answer might be a combination of different related sentences—include those too.
    - Ensure to include everything accurately required to completely and clearly answer the student's query.AIMessage
    - If any constraints are mentioned, or any point which is important to know for a particular concept include it.
    - Unless the query specifies to give only the main point, formula, law, etc. Then no need to include additional ending statements or introductory statements.

    Do NOT include:
    - General background statements
    - Sentences that are only loosely related
    - Sentences that repeat the same content with different wording

    If multiple sentences say the same thing, prefer the **clearest and most complete** one.
    If the same keyword is found in more than one place, use it only if it is clearly required for the answer (don’t keep introductions).

    {format_instructions}

    ---

    Question: {query}
    Answer: {answer}

    Candidate Sentences:
    {sentences}
    """)

    batch_size = 20
    num_batches = ceil(len(sentence_candidates) / batch_size)
    all_ids = []

    for i in range(num_batches):
        batch = sentence_candidates[i * batch_size:(i + 1) * batch_size]
        formatted_sentences = "\n".join(
            f"- [sid: {s['sid']}] {s['sentence']}" for s in batch
        )
        structured_chain = prompt | llm | parser
        try:
            result = structured_chain.invoke({
                "query": query,
                "answer": final_answer_text,   # use final two-step answer
                "sentences": formatted_sentences,
                "format_instructions": format_instructions
            })
            cleaned = []
            for tok in result.ids:
                m = re.search(r"(s\d+)", str(tok))
                if m:
                    sid = m.group(1)
                    if sid in sentence_lookup:
                        cleaned.append(sid)
            all_ids.extend(cleaned)
        except Exception as e:
            print(f"⚠️ Batch {i+1} failed: {e}")
            continue

    # ✅ Final cleanup (document order)
    keep = set(all_ids)
    returned_sids = [c["sid"] for c in sentence_candidates if c["sid"] in keep]

    # ✅ Output
    print("\n✅ Supporting Sentence IDs (merged+minimal):\n" + "=" * 50)
    print(returned_sids)

    print("\n📝 Sentences to Highlight:\n" + "=" * 50)
    for sid in returned_sids:
        entry = sentence_lookup.get(sid)
        if entry:
            print(f"{sid} (Page {entry['page']}): {entry['sentence']}")

    with open("highlight_ids.json", "w", encoding="utf-8") as f:
        json.dump(returned_sids, f, indent=2)

    return {"supporting_sentence_ids": returned_sids}

# ... keep all your existing code above ...

def build_support_ids(*, retrieved, query: str, final_answer, llm):
    """
    Backend expects this name. Calls run_file3 and returns its result.
    Writes: sentence_lookup.json, highlight_ids.json
    """
    final_answer_text = final_answer if isinstance(final_answer, str) else str(final_answer)
    return run_file3(
        retrieved=retrieved,
        query=query,
        final_answer_text=final_answer_text,
        llm=llm,
    )

__all__ = ["build_support_ids", "run_file3"]
