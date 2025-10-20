# 🗝️ Caesar Cipher Decoder

A modular Python project that automatically decodes Caesar ciphers using English letter frequencies, common words, and n-gram bonuses.  
It includes both a **CLI tool** and a **Streamlit web interface** for easy use.

---

## 🚀 Features
- Automatically detects the correct Caesar cipher shift
- Uses letter frequency, bigrams, and trigrams for improved accuracy
- Preserves case and punctuation
- Command Line Interface (CLI) for quick decoding
- Streamlit web app for an interactive GUI experience
- Comprehensive unit tests with pytest

---

## 🧩 Project Structure
```
.
├── app.py                      # Streamlit front-end GUI
├── caesar_decoder.py           # Core Caesar cipher logic
├── caesar_decoder_test.py      # Test logic in CLI 
/
    ├── test_decoder.py         # Unit tests (pytest)
├── README.md                   # Documentation
└── requirements.txt            # Dependencies
```

---

## Setup

### 1. Clone or copy this project
```bash
git clone <your-repo-url>
cd shift_decoder
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate       # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🧠 Usage

### ▶️ Run CLI
```bash
python caesar_decoder_test.py "Khoor, Zruog!"
```
**Expected output:**
```
[shift 3] Hello, World!
```

### 🧪 Run Tests
```bash
pytest -q
```
All 9 tests should pass ✅.

### 🌐 Run the Web App
```bash
streamlit run app.py
```
Then open the local URL shown in your terminal (usually http://localhost:8501).

---

## 🖱️ Using the Streamlit App
1. Paste or upload your ciphertext.
2. Click **Decode**.
3. View the best guess or all 26 candidates.
4. Optionally download the decoded text.

---

## 🧮 How It Works
- Tries all 26 possible shifts.
- Scores each result using:
  - English letter frequencies (log-likelihood)
  - Common English words
  - Common bigrams and trigrams (for short-text stability)
- Selects the plaintext with the highest combined score.

---

## 🧰 Requirements
See `requirements.txt` for details.

---

## 📄 License
MIT License — free to use, modify, and distribute.