# -------------------- secondary.py --------------------
import os
import json
import re
import unicodedata
from pathlib import Path
from typing import List
from math import ceil
from collections import defaultdict

import fitz  # PyMuPDF
import syntok.segmenter as segmenter
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Use the exact same LLM + embeddings as in your notebook cells
os.environ["GOOGLE_API_KEY"] = "AIzaSyAWAeLAPh4UoJ-aq_kOf9WbtEvCO08T2wg"
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", temperature=0, api_key=os.environ["GOOGLE_API_KEY"]
)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


def run_highlighter(
    retrieved,
    query: str,
    final_answer: str,
    pdf_in: Path,
    pdf_out: Path,
):
    """
    Runs Cell 3 → Cell 4 → Cell 5 on the retrieved chunks and writes the
    highlighted PDF to pdf_out. Prompts and logic remain exactly as in your cells.

    Returns:
      {"total_spans": int, "first_highlight_page": Optional[int]}
    """

    # =========================
    # Cell 3 — segmentation + supporting IDs (verbatim logic)
    # =========================
    sentence_candidates = []
    sentence_lookup = {}
    sid_counter = 1

    for doc in retrieved:
        chunk_uuid = doc.metadata["uuid"]
        page = doc.metadata["page"]
        source = doc.metadata["source"]
        chunk_text = doc.page_content

        paragraphs = segmenter.analyze(chunk_text)
        for paragraph in paragraphs:
            for sentence in paragraph:
                sentence_text = " ".join(token.value for token in sentence).strip()
                if not sentence_text:
                    continue

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

    with open("sentence_lookup.json", "w", encoding="utf-8") as f:
        json.dump(sentence_lookup, f, indent=2, ensure_ascii=False)

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

Do NOT include:
- General background statements
- Sentences that are only loosely related
- Sentences that repeat the same content with different wording

If multiple sentences say the same thing, prefer the **clearest and most complete** one.
If the same keyword is used is found in more than one place, use it only if it clearly is required to answer the answer accurately,
this doesn't include the introductory sentences that lead to the main point.

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
                "answer": final_answer,
                "sentences": formatted_sentences,
                "format_instructions": format_instructions
            })
            cleaned = [sid.replace("sid:", "").strip() for sid in result.ids]
            all_ids.extend(cleaned)
        except Exception as e:
            print(f"⚠️ Batch {i+1} failed: {e}")
            continue

    returned_sids = sorted(set(all_ids), key=lambda x: int(x[1:]))

    print("\n✅ Supporting Sentence IDs:\n" + "=" * 50)
    print(returned_sids)
    print("\n📝 Sentences to Highlight:\n" + "=" * 50)
    for sid in returned_sids:
        entry = sentence_lookup.get(sid)
        if entry:
            print(f"{sid} (Page {entry['page']}): {entry['sentence']}")

    with open("highlight_ids.json", "w", encoding="utf-8") as f:
        json.dump(returned_sids, f, indent=2)

    # =========================
    # Cell 4 — coverage-aware selection + safe cleaning
    # =========================
    import json as _json, re as _re
    from pathlib import Path as _Path
    from typing import List as _List
    from pydantic import BaseModel as _BaseModel
    from langchain.prompts import ChatPromptTemplate as _ChatPromptTemplate
    from langchain.output_parsers import PydanticOutputParser as _PydanticOutputParser

    base = _Path(".")
    sel_ids_path = base / "highlight_ids.json"       # from Cell 3
    lookup_path  = base / "sentence_lookup.json"     # from Cell 3
    clean_out    = base / "ai_cleaned_sentences.json"

    # ---- load Cell 3 outputs
    with open(sel_ids_path, "r", encoding="utf-8") as f:
        selected_ids = _json.load(f)

    with open(lookup_path, "r", encoding="utf-8") as f:
        sentence_lookup = _json.load(f)

    # keep original order coming from Cell 3
    candidates = []
    for sid in selected_ids:
        ent = sentence_lookup.get(sid)
        if not ent:
            continue
        candidates.append({
            "sid": sid,
            "page": int(ent["page"]),
            "sentence": ent["sentence"].strip()
        })

    # ===== PASS 1: SELECT IDS THAT COVER THE FINAL ANSWER =====
    class KeepIDs(_BaseModel):
        ids: _List[str]  # subset of provided sids, in doc order

    select_parser = _PydanticOutputParser(pydantic_object=KeepIDs)
    select_instr = select_parser.get_format_instructions()

    # Improved, subject-agnostic selection prompt
    select_prompt = ChatPromptTemplate.from_template("""
    You will choose the MINIMUM set of candidate sentences that collectively support the FINAL ANSWER.

    Inputs:
    - QUESTION: {query}
    - FINAL ANSWER: {answer}

    CANDIDATES (in document order; each line shows the page in parentheses):
    {cands}

    Selection rules:
    - Return ONLY IDs that appear in the candidate list. Preserve the same order as shown.
    - Cover EVERY major claim in the Final Answer at least once (definition/identity, key laws/rules, items in enumerations).
    - Prefer the **clearest canonical formulations** of the claim (e.g., numbered or explicitly stated laws like “(i) … (ii) …”).
    - If multiple sentences express the **same claim**, keep only ONE — choose the **earliest page number** among those equivalents.
    - EXCLUDE general reminders/introductions (e.g., “Let us recall…”, “You are already familiar…”, “Remember that…”), headings,
    labels, and page furniture.
    - For short answers (~≤120 words): keep ~2–6 sentences.
    For long answers (~≥150–300 words): keep ~6–12 sentences.

    Return ONLY this JSON:
    {format_instructions}
    """)

    def render_cands(cands):
        return "\n".join(f"- [sid: {c['sid']}] {c['sentence']}" for c in cands)

    select_chain = select_prompt | llm | select_parser
    keep = select_chain.invoke({
        "query": query,
        "answer": final_answer,
        "cands": render_cands(candidates),
        "format_instructions": select_instr
    }).ids

    # keep IDs in the original order & intersect with available
    keep_set = {sid for sid in keep if sid in {c["sid"] for c in candidates}}
    kept = [c for c in candidates if c["sid"] in keep_set]

    # ===== PASS 2: CLEAN THE KEPT SENTENCES SAFELY (NO PROSE FRACTURE) =====

    # heuristics
    BULLET_JOIN_RE = _re.compile(r"(?:\s[•●▪‣■]\s|\sz\s)", _re.I)  # split only when bullets are truly joined
    NUM_LIST_RE    = _re.compile(r"\b(?:\(?[0-9]+|[ivxlcdm]+)\)?[.)]\s+\S+", _re.I)
    TAIL_RE = _re.compile(
        r"(?i)\b(observe|write|group|classify|find out|according to|answer|project|activity|exercise|assignment|"
        r"try this|can you|let us|discuss|for example|figure|fig\.|table|chart)\b"
    )

    def looks_joined(s: str) -> bool:
        return bool(BULLET_JOIN_RE.search(s)) or (len(_re.findall(NUM_LIST_RE, s)) >= 2)

    def trim_tail(s: str) -> str:
        m = TAIL_RE.search(s)
        return s[:m.start()].rstrip(" .,:;–—-") if m else s

    def split_joined(s: str) -> List[str]:
        # turn ' z ' into ' • ' split marker without touching normal words (e.g., “Brazil”)
        t = _re.sub(r"\s+", " ", s).strip()
        t = _re.sub(r"^(?:z)\s+", "• ", t)    # start
        t = _re.sub(r"\s(z)\s", " • ", t)     # middle
        parts = [p.strip(" •-–—") for p in t.split(" • ")]
        parts = [p for p in parts if len(p.split()) >= 2]
        return parts if len(parts) >= 2 else [s.strip()]

    def strip_leading_markers(s: str) -> str:
        s = _re.sub(r"^\s*[•●▪‣■]\s*", "", s)                        # symbol bullet
        s = _re.sub(r"^\s*(?:\(?[0-9]+|[ivxlcdm]+)\)?[.)-]\s*", "", s, flags=_re.I)  # 1.  (i)  2)
        return s.strip()

    def present_in_original(orig: str, piece: str) -> bool:
        if piece in orig: return True
        def norm(x: str):
            x = _re.sub(r"\s*\(\s*", " (", x)
            x = _re.sub(r"\s*\)\s*", ")", x)
            x = _re.sub(r"\s+", " ", x)
            return x.strip()
        return norm(piece) in norm(orig)

    class CleanItem(_BaseModel):
        sid: str
        pieces: List[str]  # verbatim substrings after allowed trims (order preserved)

    class CleanBatch(_BaseModel):
        items: List[CleanItem]

    clean_parser = _PydanticOutputParser(pydantic_object=CleanBatch)
    clean_instr = clean_parser.get_format_instructions()

    clean_prompt = ChatPromptTemplate.from_template("""
    You will prepare the kept sentences for highlighting EXACTLY as they appear.

    For each entry:
    - If it is a normal prose sentence, return ONE piece (verbatim), only trimming an instruction tail if present.
    - If it is a run-on bullet line (joined by "•" / "z" / numbered items), split into separate pieces (order preserved).
    - Remove ONLY leading bullet/number markers from each piece.
    - Keep parentheses that belong to the phrase.
    - Do NOT rephrase or shorten normal sentences.
    - Ignore headings/labels (e.g., items that are just a letter/number or end with ":").

    Return ONLY this JSON:
    {format_instructions}

    Entries:
    {entries}
    """)

    # Build small cleaning batches with only kept items
    BATCH = 16
    kept_batches = [kept[i:i+BATCH] for i in range(0, len(kept), BATCH)]
    clean_results = []  # [{"sid","page","pieces":[...]}]

    for group in kept_batches:
        entries_text = "\n".join(f"- [sid: {e['sid']}] {e['sentence']}" for e in group)
        chain = clean_prompt | llm | clean_parser
        out = chain.invoke({
            "entries": entries_text,
            "format_instructions": clean_instr
        })

        # guard-rails: verify verbatim & fall back for prose
        page_map = {e["sid"]: e["page"] for e in group}
        orig_map = {e["sid"]: e["sentence"] for e in group}

        for item in out.items:
            sid = item.sid
            orig = orig_map.get(sid, "")
            page = page_map.get(sid, 1)

            # decide prose vs joined using the original text (not AI output)
            prose = not looks_joined(orig)

            if prose:
                # one piece: trim tail; if AI gave >1 or tiny fragments, fall back to full trimmed sentence
                cand = item.pieces[0] if item.pieces else orig
                cand = strip_leading_markers(trim_tail(cand)).strip() or strip_leading_markers(trim_tail(orig))
                # minimal presence/length check
                if not present_in_original(orig, cand) or len(cand.split()) < max(3, int(0.4 * len(orig.split()))):
                    cand = strip_leading_markers(trim_tail(orig))
                clean_results.append({"sid": sid, "page": page, "pieces": [cand]})
            else:
                # run-on bullets: split locally too (safety), then verify each piece
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
        _json.dump(clean_results, f, indent=2, ensure_ascii=False)

    print(f"✅ Kept {len(kept)} ids; prepared {sum(len(r['pieces']) for r in clean_results)} highlight pieces → {clean_out.name}")

    # =========================
    # Cell 5 — robust highlighter (verbatim algorithm; pdf_in/pdf_out from args)
    # =========================
    import json as __json
    import re as __re
    import unicodedata as __unicodedata
    from math import ceil as __ceil
    from pathlib import Path as __Path
    from collections import defaultdict as __defaultdict

    base_dir = __Path(".")
    _pdf_in   = __Path(pdf_in)
    _pdf_out  = __Path(pdf_out)
    clean_in = base_dir / "ai_cleaned_sentences.json"
    ids_path = base_dir / "highlight_ids.json"
    lookup_path = base_dir / "sentence_lookup.json"

    HYPHENS = r"[\-\u2010-\u2015\u2212]"
    QUOTES  = "[\u2018\u2019\u201A\u2032\u2035\u201C\u201D\u201E\u2033\u2036']"

    def fold_diacritics(s: str) -> str:
        s = __unicodedata.normalize("NFD", s)
        s = "".join(c for c in s if not __unicodedata.combining(c))
        return __unicodedata.normalize("NFC", s)

    def norm_tokenize(s: str):
        s = __unicodedata.normalize("NFKC", s).replace("\u00ad", "")
        s = fold_diacritics(s)
        s = __re.sub(HYPHENS, " ", s)
        s = __re.sub(QUOTES, " ", s)
        s = __re.sub(r"[^\w\s]", " ", s)
        s = __re.sub(r"\s+", " ", s).strip().lower()
        return s.split() if s else []

    def paren_variants(s: str):
        v1 = s
        v2 = __re.sub(r"\s*\(\s*", " (", v1)
        v2 = __re.sub(r"\s*\)\s*", ")", v2)
        v3 = __re.sub(r"[()]", "", v2)
        v3 = __re.sub(r"\s+", " ", v3).strip()
        return [v1, v2, v3]

    def build_blocks(page):
        words = page.get_text("words")
        if not words:
            return {-1: []}
        has_blocks = len(words[0]) >= 8
        blocks = __defaultdict(list)
        if has_blocks:
            for w in words:
                blocks[w[5]].append(w)
        else:
            blocks[-1] = words
        return blocks

    def stream_from_words(wlist):
        stream, tok2word = [], []
        for wi, w in enumerate(wlist):
            toks = norm_tokenize(w[4])
            for t in toks:
                stream.append(t)
                tok2word.append(wi)
        return stream, tok2word

    def rects_from_span(wlist, tok2word, i0, i1):
        word_idx_span = sorted(set(tok2word[i0:i1+1]))
        return [fitz.Rect(*wlist[idx][0:4]) for idx in word_idx_span]

    def exact_window(stream, target):
        n, m = len(stream), len(target)
        if m == 0 or n < m:
            return None
        for i in range(n - m + 1):
            if stream[i:i+m] == target:
                return (i, i+m-1)
        return None

    def tolerant_window(stream, target, max_miss=1, min_ratio=0.9):
        n, m = len(stream), len(target)
        if m == 0 or n < m:
            return None
        need = max(int(__ceil(min_ratio * m)), m - max_miss)
        best = None
        for i in range(n - m + 1):
            win = stream[i:i+m]
            matches = sum(1 for a, b in zip(win, target) if a == b)
            if matches >= need:
                best = (i, i+m-1)
                break
        return best

    # Load highlight targets
    targets = []
    if clean_in.exists():
        data = __json.loads(clean_in.read_text(encoding="utf-8"))
        for row in data:
            page = int(row["page"])
            for piece in row.get("pieces", []):
                piece = piece.strip()
                if piece:
                    targets.append({"page": page, "text": piece})
    else:
        ids = __json.loads(ids_path.read_text(encoding="utf-8"))
        lookup = __json.loads(lookup_path.read_text(encoding="utf-8"))
        for sid in ids:
            ent = lookup.get(sid)
            if ent:
                targets.append({"page": int(ent["page"]), "text": ent["sentence"]})

    by_page = __defaultdict(list)
    for t in targets:
        by_page[int(t["page"])].append(t["text"])

    # Highlight, track which pages actually got spans
    doc = fitz.open(str(_pdf_in))
    total_spans, misses = 0, []
    pages_with_spans = set()

    for page_num, pieces in sorted(by_page.items()):
        page = doc[page_num - 1]
        blocks = build_blocks(page)

        for raw in pieces:
            found = False
            target_tokens = norm_tokenize(raw)
            if not target_tokens:
                continue

            # 1) block-scoped exact
            for _, wlist in blocks.items():
                stream, tok2word = stream_from_words(wlist)
                loc = exact_window(stream, target_tokens)
                if loc:
                    rlist = rects_from_span(wlist, tok2word, *loc)
                    for r in rlist:
                        page.add_highlight_annot(r)
                        total_spans += 1
                        pages_with_spans.add(page_num)
                    found = True
                    break
            if found:
                continue

            # 2) page-wide exact
            all_words = [w for bl in blocks.values() for w in bl]
            stream, tok2word = stream_from_words(all_words)
            loc = exact_window(stream, target_tokens)
            if loc:
                rlist = rects_from_span(all_words, tok2word, *loc)
                for r in rlist:
                    page.add_highlight_annot(r)
                    total_spans += 1
                    pages_with_spans.add(page_num)
                continue

            # 3) search_for variants
            for variant in paren_variants(raw):
                v = unicodedata.normalize("NFKC", variant).replace("\u00ad", "")
                v = fold_diacritics(v)
                v = re.sub(HYPHENS, " ", v)
                v = re.sub(QUOTES, " ", v)
                v = re.sub(r"\s+", " ", v).strip()
                if not v:
                    continue
                quads = page.search_for(v, quads=True)
                if quads:
                    for q in quads:
                        page.add_highlight_annot(q)
                        total_spans += 1
                        pages_with_spans.add(page_num)
                    found = True
                    break
            if found:
                continue

            # 4) tolerant window
            if len(target_tokens) >= 10:
                loc = tolerant_window(stream, target_tokens, max_miss=1, min_ratio=0.9)
                if loc:
                    rlist = rects_from_span(all_words, tok2word, *loc)
                    for r in rlist:
                        page.add_highlight_annot(r)
                        total_spans += 1
                        pages_with_spans.add(page_num)
                    found = True

            if not found:
                misses.append(raw[:90])

    doc.save(str(_pdf_out), garbage=4, deflate=True)
    doc.close()

    print(f"✅ Highlighted {total_spans} spans → {_pdf_out}")
    if misses:
        print("⚠️ Not found (sample):", misses[:6])

    first_highlight_page = min(pages_with_spans) if pages_with_spans else None
    return {"total_spans": total_spans, "first_highlight_page": first_highlight_page}
