# Unit tests for checks of modules

import caesar_decoder as dec

def test_cipher_roundtrip_all_shifts():
    plain = "The quick brown fox jumps over the lazy dog."
    for k in range(26):
        cipher = dec.caesar_shift(plain, k)
        back = dec.caesar_shift(cipher, -k)
        assert back == plain

def test_wraparound_lower():
    # xyz shifted forward by 4 -> bcd
    assert dec.caesar_shift("xyz", 4) == "bcd"

def test_wraparound_upper():
    # XYZ shifted forward by 4 -> BCD
    assert dec.caesar_shift("XYZ", 4) == "BCD"

def test_find_best_hello_world():
    best = dec.find_best("Khoor, Zruog!")
    assert best["shift"] == 3
    assert best["plaintext"].startswith("Hello, World!")

def test_decrypt_all_returns_26():
    res = dec.decrypt_all("Khoor, Zruog!")
    assert len(res) == 26

def test_non_ascii_passthrough_known_shift():
    # Use the transformation directly for non-ASCII stability
    plain = "Hello, World! Café."
    cipher = dec.caesar_shift(plain, 3)
    assert dec.caesar_shift(cipher, -3) == plain

def test_short_text_avoid_ambiguous_find_best():
    # For very short strings, detector can be ambiguous; test transform, not ranking
    assert dec.caesar_shift("bcd", -4) == "xyz"
    assert dec.caesar_shift("BCD", -4) == "XYZ"

def test_detector_on_sentence():
    # A longer, Englishy sentence so the auto-detector is stable
    cipher = "Uifsf jt op tqppo."
    best = dec.find_best(cipher)
    assert best["shift"] == 1
    assert "There is no spoon." in best["plaintext"]

def test_property_pangram_all_keys():
    base = "Sphinx of black quartz, judge my vow."
    for k in range(26):
        cipher = dec.caesar_shift(base, k)
        best = dec.find_best(cipher)
        # On a long pangram, detector should recover exact shift & plaintext
        assert best["shift"] == k
        assert best["plaintext"] == base
