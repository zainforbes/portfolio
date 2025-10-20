# CLI tool!
# Python 3.8+

# TEST BY RUNNING python caesar_decoder_test.py "example text"

import sys, argparse, json, math, string
from collections import Counter

# Approximate English letter frequencies (A..Z), normalized to probabilities.
EXPECTED_FREQ = {
    'a': 0.082, 'b': 0.015, 'c': 0.028, 'd': 0.043, 'e': 0.127, 'f': 0.022,
    'g': 0.020, 'h': 0.061, 'i': 0.070, 'j': 0.002, 'k': 0.008, 'l': 0.040,
    'm': 0.024, 'n': 0.067, 'o': 0.075, 'p': 0.019, 'q': 0.001, 'r': 0.060,
    's': 0.063, 't': 0.091, 'u': 0.028, 'v': 0.010, 'w': 0.024, 'x': 0.002,
    'y': 0.020, 'z': 0.001
}
# Common words to stabilize ranking on short texts
COMMON_WORDS = {"the","and","to","of","in","is","that","for","with","on","we","you","it","as","be"}

def caesar_shift(text: str, shift: int) -> str:
    """Shift ASCII letters by 'shift' (mod 26); preserve case & punctuation."""
    s = shift % 26
    out = []
    for c in text:
        if 'a' <= c <= 'z':
            out.append(chr((ord(c) - 97 + s) % 26 + 97))
        elif 'A' <= c <= 'Z':
            out.append(chr((ord(c) - 65 + s) % 26 + 65))
        else:
            out.append(c)
    return ''.join(out)

def english_loglikelihood(text: str) -> float:
    """
    Log-likelihood under English unigram model.
    Higher is better. Works well even for short texts.
    """
    letters = [c for c in text.lower() if 'a' <= c <= 'z']
    if not letters:
        return float('-inf')
    counts = Counter(letters)
    eps = 1e-9  # avoid log(0)
    return sum(counts[ch] * math.log(EXPECTED_FREQ.get(ch, eps)) for ch in string.ascii_lowercase)

def word_hits(text: str) -> int:
    tokens = "".join(ch if ch.isalpha() else " " for ch in text.lower()).split()
    return sum(t in COMMON_WORDS for t in tokens)

def decrypt_all(ciphertext: str):
    """Return list of dicts: shift, plaintext, score (sorted best→worst)."""
    results = []
    for k in range(26):
        # decrypt means shifting BACK by k
        pt = caesar_shift(ciphertext, -k)
        score = english_loglikelihood(pt) + 1.5 * word_hits(pt)
        results.append({"shift": k, "plaintext": pt, "score": score})
    results.sort(key=lambda r: (-r["score"], r["shift"]))  # best score first; tie → lower shift
    return results

def find_best(ciphertext: str):
    return decrypt_all(ciphertext)[0]

def main():
    ap = argparse.ArgumentParser(description="English-only Caesar cipher decoder (auto-shift).")
    ap.add_argument("ciphertext", nargs="?", help="Ciphertext. If omitted, read from stdin.")
    ap.add_argument("--all", action="store_true", help="Print all 26 candidates.")
    ap.add_argument("--shift", type=int, help="Decode with known shift (0..25).")
    ap.add_argument("--json", action="store_true", help="JSON output (machine-friendly).")
    ap.add_argument("--selftest", action="store_true", help="Run quick self-tests and exit.")
    args = ap.parse_args()

    if args.selftest:
        assert caesar_shift("Abc-Z", 2) == "Cde-B"
        # "Hello, World!" encrypted with shift +3 is "Khoor, Zruog!"
        assert find_best("Khoor, Zruog!")["shift"] == 3
        print("Self-test OK"); return

    text = args.ciphertext if args.ciphertext is not None else sys.stdin.read()

    if args.shift is not None:
        decoded = caesar_shift(text, -args.shift)
        if args.json:
            print(json.dumps({"shift": args.shift % 26, "plaintext": decoded}, ensure_ascii=False))
        else:
            print(decoded)
        return

    if args.all:
        results = decrypt_all(text)
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            for r in results:
                print(f"[shift {r['shift']:2d}] {r['plaintext']}")
        return

    best = find_best(text)
    if args.json:
        print(json.dumps(best, ensure_ascii=False, indent=2))
    else:
        print(f"[shift {best['shift']:2d}] {best['plaintext']}")

if __name__ == "__main__":
    main()

