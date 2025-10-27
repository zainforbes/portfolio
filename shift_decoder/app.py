import streamlit as st
from caesar_decoder import decrypt_all, find_best, caesar_shift

st.set_page_config(page_title="Caesar Decoder", page_icon=" ", layout="wide")
st.title("Caesar Cipher Decoder (English)")

col1, col2 = st.columns([2, 1])

with col1:
    text = st.text_area("Paste your ciphertext here", height=220)
    uploaded = st.file_uploader("…or upload a .txt file", type=["txt"])
    if uploaded:
        text = uploaded.read().decode("utf-8", errors="ignore")

with col2:
    known = st.checkbox("I know the shift key")
    k = st.slider("Shift", 0, 25, 0, disabled=not known)
    show_all = st.checkbox("Show all 26 candidates")
    run = st.button("Decode")

if run and text:
    if known:
        plaintext = caesar_shift(text, -k)
        st.subheader(f"Decoded (shift {k})")
        st.code(plaintext)
        st.download_button("Download decoded text", plaintext, file_name=f"decoded_shift_{k}.txt")
    else:
        results = decrypt_all(text)
        best = find_best(text)
        st.subheader(f"Best guess — shift {best['shift']}")
        st.code(best["plaintext"])
        st.download_button("Download best guess", best["plaintext"], file_name=f"decoded_shift_{best['shift']}.txt")

        if show_all:
            st.markdown("### All candidates")
            for r in results:
                with st.expander(f"Shift {r['shift']} (score {r['score']:.2f})"):
                    st.code(r["plaintext"])
