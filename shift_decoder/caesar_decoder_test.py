# CLI TOOL 
import sys, argparse, json
import caesar_decoder as dec 

def main():
    ap = argparse.ArgumentParser(description="English-only Caesar cipher decoder (auto-shift).")
    ap.add_argument("ciphertext", nargs="?", help="Ciphertext. If omitted, read from stdin.")
    ap.add_argument("--all", action="store_true", help="Print all 26 candidates.")
    ap.add_argument("--shift", type=int, help="Decode with known shift (0..25).")
    ap.add_argument("--json", action="store_true", help="JSON output (machine-friendly).")
    ap.add_argument("--selftest", action="store_true", help="Run quick self-tests and exit.")
    args = ap.parse_args()

    if args.selftest:
        assert dec.caesar_shift("Abc-Z", 2) == "Cde-B"
        assert dec.find_best("Khoor, Zruog!")["shift"] == 3
        print("Self-test OK"); return

    text = args.ciphertext if args.ciphertext is not None else sys.stdin.read()

    if args.shift is not None:
        decoded = dec.caesar_shift(text, -args.shift)
        print(json.dumps({"shift": args.shift % 26, "plaintext": decoded}, ensure_ascii=False)
              if args.json else decoded)
        return

    if args.all:
        results = dec.decrypt_all(text)
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            for r in results:
                print(f"[shift {r['shift']:2d}] {r['plaintext']}")
        return

    best = dec.find_best(text)
    print(json.dumps(best, ensure_ascii=False, indent=2)
          if args.json else f"[shift {best['shift']:2d}] {best['plaintext']}")

if __name__ == "__main__":
    main()
