"""
ner_preprocess_only.py
Runs spaCy NER on a paper and outputs a cleaned text file
ready to paste into Claude.ai for triple extraction.

pip install spacy pdfminer.six
python -m spacy download en_core_web_lg
python ner_preprocess_only.py
"""

import re
import sys
from pathlib import Path
import spacy

SCRIPT_DIR  = Path(__file__).parent
INPUT_PATH  = SCRIPT_DIR / "papers" / "FAIA-408-FAIA250695.pdf"
OUTPUT_PATH = SCRIPT_DIR / "ner_output" / "06495.txt"


# ─────────────────────────────────────────────────────────────
SPACY_MODEL    = "en_core_web_lg"
KEEP_ENT_TYPES = {
    "ORG", "PRODUCT", "WORK_OF_ART", "LAW",
    "GPE", "LOC", "PERSON", "NORP", "FAC", "EVENT",
}


def to_upper_camel(text: str) -> str:
    """Convert any string to UpperCamelCase."""
    clean = re.sub(r"[^\w\s\-]", "", text)
    words = re.split(r"[\s\-_]+", clean.strip())
    return "".join(w.capitalize() for w in words if w)


def load_text(path: str) -> str:
    """Load plain text or PDF."""
    p = Path(path)
    if not p.exists():
        print(f"Error: file not found: {path}")
        sys.exit(1)
    if p.suffix.lower() == ".pdf":
        try:
            from pdfminer.high_level import extract_text
            return extract_text(path)
        except ImportError:
            print("Install pdfminer.six: pip install pdfminer.six")
            sys.exit(1)
    return p.read_text(encoding="utf-8")


def run_ner(text: str, nlp) -> dict[str, str]:
    """Run spaCy NER and return surface form -> canonical label mapping."""
    doc = nlp(text)
    entity_map = {}
    for ent in doc.ents:
        if ent.label_ not in KEEP_ENT_TYPES:
            continue
        surface   = ent.text.strip()
        canonical = to_upper_camel(surface)
        if len(canonical) >= 3:
            entity_map[surface] = canonical
    return entity_map


def substitute_entities(text: str, entity_map: dict[str, str]) -> str:
    """Replace surface forms with canonical labels, longest first."""
    for surface, canonical in sorted(entity_map.items(), key=lambda x: -len(x[0])):
        text = re.sub(r"\b" + re.escape(surface) + r"\b", canonical, text)
    return text


def compute_metrics(original: str, cleaned: str, entity_map: dict) -> None:
    """Print a short summary of what changed."""
    orig_words    = len(original.split())
    cleaned_words = len(cleaned.split())
    print(f"\n  Entities recognised and canonicalised : {len(entity_map)}")
    print(f"  Word count before                     : {orig_words}")
    print(f"  Word count after                      : {cleaned_words}")
    if entity_map:
        print("\n  Sample mappings (first 8):")
        for surface, canonical in list(entity_map.items())[:8]:
            print(f"    '{surface}' -> '{canonical}'")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(INPUT_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    original_text = load_text(args.input)
    print(f"  {len(original_text)} characters loaded")

    print(f"Loading spaCy ({SPACY_MODEL})...")
    nlp = spacy.load(SPACY_MODEL)

    print("Running NER...")
    entity_map = run_ner(original_text, nlp)

    print("Substituting canonical labels...")
    cleaned_text = substitute_entities(original_text, entity_map)

    compute_metrics(original_text, cleaned_text, entity_map)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cleaned_text, encoding="utf-8")
    print(f"\nCleaned text written to: {output_path}")


if __name__ == "__main__":
    main()