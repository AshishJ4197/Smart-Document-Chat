# -------------------- file5.py (Cell 5 wrapped) --------------------
# robust highlighter (token-aware + tolerant + scoped symbol highlights)
import fitz  # PyMuPDF
import json
import re
import unicodedata
from math import ceil
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List

# ---------- normalization / tokenization

# Hyphens to normalize to spaces
HYPHENS = r"[\-\u2010-\u2015\u2212]"  # -, ‐, –, —, −

# IMPORTANT: do NOT include Unicode primes here; we want to preserve them.
# Keep only typographic quotes/apostrophes.
QUOTES  = r"[\u2018\u2019\u201A\u201C\u201D\u201E']"

# Unicode prime characters (′ ″ ‴ ⁗)
PRIME_CHARS = "\u2032\u2033\u2034\u2057"

def fold_diacritics(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", s)

def normalize_primes_to_tokens(s: str) -> str:
    # Map primes to textual tokens so token windows align even when PDF splits them.
    # e.g., "h′" → "h prime", "″" → " doubleprime"
    s = s.replace("\u2032", " prime")
    s = s.replace("\u2033", " doubleprime")
    s = s.replace("\u2034", " tripleprime")
    s = s.replace("\u2057", " quadrupleprime")
    return s

def prime_variants(s: str):
    # Return variants that keep primes intact or map them to ASCII apostrophes / remove.
    # Used by search_for to increase robustness.
    v = s
    # 1) raw (keep exact)
    yield v
    # 2) map primes to ASCII apostrophe
    v2 = v.translate({ord("\u2032"): ord("'"), ord("\u2033"): ord('"')})
    yield v2
    # 3) drop primes (sometimes PDF glues them to letters awkwardly)
    v3 = re.sub(f"[{PRIME_CHARS}]", "", v)
    yield v3

def norm_tokenize(s: str) -> List[str]:
    # tokenization that preserves semantic symbols:
    # - keep primes by mapping them to tokens (prime/doubleprime/…)
    # - keep slashes for ratios via later operator highlighting; for token stream, we drop punctuation
    s = unicodedata.normalize("NFKC", s).replace("\u00ad", "")
    s = fold_diacritics(s)
    s = normalize_primes_to_tokens(s)            # <-- preserve as words
    s = re.sub(HYPHENS, " ", s)
    s = re.sub(QUOTES, " ", s)
    # Keep all other punctuation out of tokens
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s.split() if s else []

def paren_variants(s: str) -> List[str]:
    # Variants for parentheses spacing; keep primes intact
    v1 = s
    v2 = re.sub(r"\s*\(\s*", " (", v1)
    v2 = re.sub(r"\s*\)\s*", ")", v2)
    v3 = re.sub(r"[()]", "", v2)
    v3 = re.sub(r"\s+", " ", v3).strip()
    return [v1, v2, v3]

# ---------- operator/symbol support (scoped to matched sentence region)
OP_CHARS = (
    "=<>±≈≠≤≥∝↔→"          # relations
    "+−-*/^×·÷"            # arithmetic (ASCII '-' and U+2212 '−')
    "√∞∑Σ∫πθΔ∂µΩ°"
)

# Also treat parentheses and primes as micro-highlights inside region
EXTRA_REGION_SYMS = "()"
PRIME_SET = set(PRIME_CHARS)

OP_STRINGS = ["<=", ">=", "==", "!=", "+=", "-=", "->", "<-", "<->"]

def has_ops_or_primes_or_parens(s: str) -> bool:
    return (
        any(ch in s for ch in OP_CHARS)
        or any(seq in s for seq in OP_STRINGS)
        or any(ch in s for ch in EXTRA_REGION_SYMS)
        or any(ch in s for ch in PRIME_SET)
    )

# ---------- utilities
def build_blocks(page):
    """Return {block_idx: [word tuples]} using page.get_text('words')."""
    words = page.get_text("words")  # (x0,y0,x1,y1,text,block_no,line_no,word_no)
    if not words:
        return {-1: []}
    has_blocks = len(words[0]) >= 8
    blocks = defaultdict(list)
    if has_blocks:
        for w in words:
            blocks[w[5]].append(w)
    else:
        blocks[-1] = words
    return blocks

def stream_from_words(wlist):
    stream, tok2word = [], []
    for wi, w in enumerate(wlist):
        toks = norm_tokenize(w[4])    # tokenizing PDF word text with same rules
        for t in toks:
            stream.append(t)
            tok2word.append(wi)
    return stream, tok2word

def rects_from_span(wlist, tok2word, i0, i1):
    idxs = sorted(set(tok2word[i0:i1+1]))
    return [fitz.Rect(*wlist[idx][0:4]) for idx in idxs]

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
    need = max(int(ceil(min_ratio * m)), m - max_miss)
    for i in range(n - m + 1):
        win = stream[i:i+m]
        matches = sum(1 for a, b in zip(win, target) if a == b)
        if matches >= need:
            return (i, i+m-1)
    return None

def union_rect(rects):
    if not rects:
        return None
    u = fitz.Rect(rects[0])
    for r in rects[1:]:
        u |= r
    return u

def run_cell5(
    pdf_in_path: str,
    clean_in_path: str = "ai_cleaned_sentences.json",   # from Cell 4
    ids_path: str = "highlight_ids.json",               # fallback if Cell 4 is skipped
    lookup_path: str = "sentence_lookup.json",          # fallback
    pdf_out_path: str = "highlighted_output.pdf",
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    pdf_in_path : str
        Path to the uploaded PDF to annotate.
    clean_in_path : str
        Path to cleaned pieces (from file4 / Cell 4). If missing, falls back to ids+lookup.
    ids_path : str
        Path to highlight_ids.json (from file3 / Cell 3).
    lookup_path : str
        Path to sentence_lookup.json (from file3 / Cell 3).
    pdf_out_path : str
        Output annotated PDF path.

    Returns
    -------
    dict with keys:
        total_spans, misses_sample, pdf_out_path
    """
    base_dir = Path(".")
    pdf_in   = base_dir / pdf_in_path
    pdf_out  = base_dir / pdf_out_path
    clean_in = base_dir / clean_in_path
    ids_pth  = base_dir / ids_path
    lookup_p = base_dir / lookup_path

    # ---------- load targets
    targets = []  # {"page": int, "text": str}

    if clean_in.exists():
        data = json.loads(clean_in.read_text(encoding="utf-8"))
        for row in data:
            page = int(row["page"])
            for piece in row.get("pieces", []):
                piece = piece.strip()
                if piece:
                    targets.append({"page": page, "text": piece})
    else:
        # fallback to raw ids
        ids = json.loads(ids_pth.read_text(encoding="utf-8"))
        lookup = json.loads(lookup_p.read_text(encoding="utf-8"))
        for sid in ids:
            ent = lookup.get(sid)
            if ent:
                targets.append({"page": int(ent["page"]), "text": ent["sentence"]})

    # group targets per page
    by_page = defaultdict(list)
    for t in targets:
        by_page[int(t["page"])].append(t["text"])

    # ---------- highlight
    doc = fitz.open(pdf_in)
    total_spans, misses = 0, []

    for page_num, pieces in sorted(by_page.items()):
        page = doc[page_num - 1]
        blocks = build_blocks(page)

        for raw in pieces:
            if not raw.strip():
                continue

            found = False
            sentence_rects = None
            target_tokens = norm_tokenize(raw)
            if not target_tokens:
                continue

            # 1) block-scoped exact token window
            for _b_idx, wlist in blocks.items():
                stream, tok2word = stream_from_words(wlist)
                loc = exact_window(stream, target_tokens)
                if loc:
                    rlist = rects_from_span(wlist, tok2word, *loc)
                    for r in rlist:
                        page.add_highlight_annot(r)
                        total_spans += 1
                    sentence_rects = rlist
                    found = True
                    break

            # 2) page-wide exact token window
            if not found:
                all_words = [w for bl in blocks.values() for w in bl]
                stream, tok2word = stream_from_words(all_words)
                loc = exact_window(stream, target_tokens)
                if loc:
                    rlist = rects_from_span(all_words, tok2word, *loc)
                    for r in rlist:
                        page.add_highlight_annot(r)
                        total_spans += 1
                    sentence_rects = rlist
                    found = True

            # 3) search_for with paren/prime variants (no stripping of primes!)
            if not found:
                for base in paren_variants(raw):
                    for variant in prime_variants(base):
                        v = unicodedata.normalize("NFKC", variant).replace("\u00ad", "")
                        v = fold_diacritics(v)
                        # normalize hyphens/quotes spacing but DO NOT remove primes
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
                            sentence_rects = [fitz.Rect(q.rect) for q in quads]
                            found = True
                            break
                    if found:
                        break

            # 4) tolerant page-wide token window (only for longer phrases)
            if not found and len(target_tokens) >= 10:
                all_words = [w for bl in blocks.values() for w in bl]
                stream, tok2word = stream_from_words(all_words)
                loc = tolerant_window(stream, target_tokens, max_miss=1, min_ratio=0.9)
                if loc:
                    rlist = rects_from_span(all_words, tok2word, *loc)
                    for r in rlist:
                        page.add_highlight_annot(r)
                        total_spans += 1
                    sentence_rects = rlist
                    found = True

            # ----- scoped micro-highlights: operators + primes + parentheses -----
            if found and sentence_rects and has_ops_or_primes_or_parens(raw):
                region = union_rect(sentence_rects)
                pad = 1.5
                region = fitz.Rect(region.x0 - pad, region.y0 - pad, region.x1 + pad, region.y1 + pad)

                # single-glyph operators
                for ch in set(OP_CHARS):
                    if ch in raw:
                        quads = page.search_for(ch, quads=True)
                        for q in quads:
                            r = fitz.Rect(q.rect)
                            if region.contains(r):
                                page.add_highlight_annot(q)
                                total_spans += 1

                # multi-char ASCII sequences
                for seq in OP_STRINGS:
                    if seq in raw:
                        quads = page.search_for(seq, quads=True)
                        for q in quads:
                            r = fitz.Rect(q.rect)
                            if region.contains(r):
                                page.add_highlight_annot(q)
                                total_spans += 1

                # parentheses
                for ch in "()":
                    if ch in raw:
                        quads = page.search_for(ch, quads=True)
                        for q in quads:
                            r = fitz.Rect(q.rect)
                            if region.contains(r):
                                page.add_highlight_annot(q)
                                total_spans += 1

                # primes (′ ″ ‴ ⁗)
                for ch in PRIME_SET:
                    if ch in raw:
                        quads = page.search_for(ch, quads=True)
                        for q in quads:
                            r = fitz.Rect(q.rect)
                            if region.contains(r):
                                page.add_highlight_annot(q)
                                total_spans += 1

            if not found:
                misses.append(raw[:90])

    doc.save(pdf_out, garbage=4, deflate=True)
    doc.close()

    return {
        "total_spans": total_spans,
        "misses_sample": misses[:6],
        "pdf_out_path": str(pdf_out),
    }

# ... keep all your existing code above ...

def highlight_pdf(*, pdf_in, pdf_out):
    """
    Backend expects this name. Calls run_cell5 and returns its result.
    Reads:  ai_cleaned_sentences.json (or highlight_ids.json + sentence_lookup.json)
    Writes: highlighted_output.pdf (to pdf_out)
    """
    return run_cell5(
        pdf_in_path=str(pdf_in),
        clean_in_path="ai_cleaned_sentences.json",
        ids_path="highlight_ids.json",
        lookup_path="sentence_lookup.json",
        pdf_out_path=str(pdf_out),
    )

__all__ = ["highlight_pdf", "run_cell5"]
