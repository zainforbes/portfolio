import gradio as gr
from tools import qa_with_context

def chat_fn(user_input, history=[]):
    answer = qa_with_context(user_input)
    history.append((user_input, answer))
    return history, history

with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Chat with Your CV")
    chatbot = gr.Chatbot()
    txt = gr.Textbox(placeholder="Ask me about my background…")
    txt.submit(chat_fn, [txt, chatbot], [chatbot, chatbot])
    demo.launch()
