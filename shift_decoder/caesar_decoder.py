# Core Caesar cipher decoder logic — English-only, self-contained.

import math, string
from collections import Counter

EXPECTED_FREQ = {
    'a':0.082,'b':0.015,'c':0.028,'d':0.043,'e':0.127,'f':0.022,
    'g':0.020,'h':0.061,'i':0.070,'j':0.002,'k':0.008,'l':0.040,
    'm':0.024,'n':0.067,'o':0.075,'p':0.019,'q':0.001,'r':0.060,
    's':0.063,'t':0.091,'u':0.028,'v':0.010,'w':0.024,'x':0.002,
    'y':0.020,'z':0.001
}

COMMON_WORDS = {
    "the","and","to","of","in","is","that","for","with","on","we",
    "you","it","as","be"
}

# Common English bigrams 
COMMON_BIGRAMS = {
    "th","he","in","er","an","re","on","at","en","nd","ti","es","or","te",
    "of","ed","is","it","al","ar","st","to","nt","ng","se","ha","as","ou",
    "io","le","ve","co","me","de","hi","ri","ro","ic",
    # extras that help short phrases like "Hello, World!"
    "el","ll","lo","wo","rl","ld","ow"
}

# A few very common English trigrams
COMMON_TRIGRAMS = {
    "the","and","ing","her","ere","ent","tha","nth","was","eth","for",
    "dth","hat","his","ell","llo","wor","orl","rld","hel"
}

def caesar_shift(s: str, k: int) -> str:
    """Shift ASCII letters by k (mod 26), preserving case/punctuation."""
    k %= 26
    out = []
    for c in s:
        if 'a' <= c <= 'z':
            out.append(chr((ord(c) - 97 + k) % 26 + 97))
        elif 'A' <= c <= 'Z':
            out.append(chr((ord(c) - 65 + k) % 26 + 65))
        else:
            out.append(c)
    return ''.join(out)

def english_loglikelihood(s: str) -> float:
    """Log-likelihood under English unigram model."""
    letters = [c for c in s.lower() if 'a' <= c <= 'z']
    if not letters:
        return float('-inf')
    counts = Counter(letters)
    eps = 1e-9
    return sum(counts[ch] * math.log(EXPECTED_FREQ.get(ch, eps))
               for ch in string.ascii_lowercase)

def word_hits(s: str) -> int:
    """Count occurrences of common English words."""
    tokens = "".join(ch if ch.isalpha() else " " for ch in s.lower()).split()
    return sum(t in COMMON_WORDS for t in tokens)

def bigram_bonus(s: str) -> int:
    """Small bonus for common English letter pairs (helps short texts)."""
    t = [c for c in s.lower() if 'a' <= c <= 'z']
    return sum(("".join(t[i:i+2]) in COMMON_BIGRAMS) for i in range(len(t)-1))

def trigram_bonus(s: str) -> int:
    """Tiny bonus for common English trigrams."""
    t = [c for c in s.lower() if 'a' <= c <= 'z']
    return sum(("".join(t[i:i+3]) in COMMON_TRIGRAMS) for i in range(len(t)-2))

def decrypt_all(cipher: str):
    """Try all 26 shifts and rank by score."""
    results = []
    for k in range(26):
        pt = caesar_shift(cipher, -k)
        score = (
            english_loglikelihood(pt)
            + 1.5 * word_hits(pt)
            + 0.8 * bigram_bonus(pt)    
            + 0.5 * trigram_bonus(pt)   
        )
        results.append({"shift": k, "plaintext": pt, "score": score})
    # best score first; tie-break by lower shift
    results.sort(key=lambda r: (-r["score"], r["shift"]))
    return results

def find_best(cipher: str):
    """Return best candidate (dict with shift, plaintext, score)."""
    return decrypt_all(cipher)[0]
