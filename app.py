import os
import gradio as gr
import PyPDF2
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
# ----------------------------------------------------------------
# 1. Model Configuration
# ----------------------------------------------------------------
LLAMA_MODEL_NAME = (
    "/root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-70B-Instruct"
    "/snapshots/945c8663693130f8be2ee66210e062158b2a9693"
)

print("Loading Llama model, please wait...")
tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME)
#quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    LLAMA_MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
    #quantization_config=quantization_config
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False
)

# ----------------------------------------------------------------
# 2. PDF text extraction function
# ----------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        with open(pdf_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        return f"[Error] An error occurred while reading the PDF: {str(e)}"

# ----------------------------------------------------------------
# 3. Merge messages to prompt function
# ----------------------------------------------------------------
def merge_messages_to_prompt(messages):
    """
    Concatenate a list of messages of the form:
    [
      {"role": "system",    "content": "..."},
      {"role": "user",      "content": "..."},
      {"role": "assistant", "content": "..."},
      ...
    ]
    into a single prompt string.
    """
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt += f"System: {content}\n\n"
        elif role == "user":
            prompt += f"User: {content}\n\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n\n"
    # Reserve a final "Assistant:" prompt
    prompt += "Assistant:"
    return prompt

# ----------------------------------------------------------------
# 4. Two-step conversation update logic
# ----------------------------------------------------------------
def add_user_message(user_input, chat_history):
    """
    Step 1: Immediately add the new user message to the conversation and clear the input box.
    """
    chat_history.append((user_input, ""))  # Append user message; leave AI reply empty for now
    return chat_history, chat_history, ""

def generate_assistant_reply(pdf_content, chat_history):
    """
    Step 2: Use only the latest user message, call the model to generate an answer, and update the conversation.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a chatbot who always responds accurately.\n"
                "Below is the PDF content (if any):\n"
                f"{pdf_content}\n"
            )
        }
    ]
    # Use only the latest user message
    last_user_message, _ = chat_history[-1]
    messages.append({"role": "user", "content": last_user_message})
    
    prompt = merge_messages_to_prompt(messages)
    outputs = generator(
        prompt,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.7
    )
    reply_text = outputs[0]["generated_text"].strip()

    if reply_text.startswith("Assistant:"):
        reply_text = reply_text[len("Assistant:"):].strip()

    last_user, _ = chat_history[-1]
    chat_history[-1] = (last_user, reply_text)
    return chat_history, chat_history

def update_pdf_content(pdf_file):
    """
    When a PDF is uploaded by the user, update the pdf_content state.
    """
    if pdf_file is None:
        return ""
    pdf_text = extract_text_from_pdf(pdf_file)
    if pdf_text.startswith("[Error]"):
        return ""
    if len(pdf_text.strip()) == 0:
        return "No valid text was extracted from the PDF. Please check the file."
    return pdf_text

# ----------------------------------------------------------------
# 5. Frontend UI + CSS
# ----------------------------------------------------------------
custom_css = """
body {
    background: linear-gradient(to top, #ADD8E6, #ffffff) !important;
    color: #000 !important;
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    margin: 0;
    padding: 0;
}
footer {display: none !important;}
.gradio-container {
    background: linear-gradient(to top, #ADD8E6, #ffffff) !important;
    width: 100% !important;
    max-width: none !important;
    margin: 0 !important;
    padding: 2rem !important;
    min-height: 100vh !important;
    box-sizing: border-box !important;
}
#title {
    font-size: 3.5rem;
    font-weight: 600;
    margin-bottom: 1rem;
    text-align: left;
}
#subtitle {
    font-size: 1.6rem;
    color: #555;
    margin-bottom: 1.5rem;
    text-align: left;
}
/* Set Chatbot area to be responsive */
#chatbot {
    width: 100% !important;
    min-height: 600px !important;
}
.gr-chat-message {
    border-radius: 8px !important;
    margin-bottom: 8px !important;
    padding: 8px !important;
    font-size: 1.5rem !important;
}
.gr-chat-message.user .gr-chat-message-text {
    background-color: #FFFFFF !important;
    color: #000 !important;
}
.gr-chat-message.bot .gr-chat-message-text {
    background-color: #FFFFFF !important;
    color: #000 !important;
}
.send-btn {
    background-color: #000000 !important;
    color: #FFFFFF !important;
}
@media (max-width: 600px) {
    .gradio-container {
        padding: 1rem !important;
    }
    #title {
        font-size: 2.5rem;
    }
    #subtitle {
        font-size: 1.4rem;
    }
    .gr-chat-message {
        font-size: 1.4rem !important;
    }
}
.gradio-container label {
    font-size: 1.4rem !important;
    font-weight: 700 !important;
}

.send-btn {
    font-size: 1.4rem !important;
    font-weight: 700 !important;
}

.gr-textbox textarea::placeholder {
    font-size: 1.4rem !important;
    font-weight: 700 !important;
}
.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: #444 !important; 
}

"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<div id='title'> PolicyPilot</div>", elem_id="title")
    gr.Markdown(
    "<div id='subtitle'><b>Please provide a scientific paper for PolicyPilot to generate a detailed, accurate policy brief.</b></div>",
    elem_id="subtitle"
)

    pdf_content_state = gr.State("")
    chat_history_state = gr.State([])

    # Add elem_id to Chatbot for CSS customization
    chatbot = gr.Chatbot(label="Chatbot", height=600, elem_id="chatbot")
    
    with gr.Row():
        pdf_file_input = gr.File(
            label="Upload PDF (optional)",
            file_types=[".pdf"],
            type="filepath",
            scale=1
        )
        user_input = gr.Textbox(
            placeholder="Type your question here...",
            #label="",
            show_label=False,
            scale=3
        )
        send_btn = gr.Button(
            "Send",
            variant="primary",
            elem_classes=["send-btn"],
            scale=1
        )

    # Disable queuing
    pdf_file_input.change(
        fn=update_pdf_content,
        inputs=[pdf_file_input],
        outputs=[pdf_content_state],
        queue=False
    )

    # Send button - Two-step callback
    send_btn.click(
        fn=add_user_message,
        inputs=[user_input, chat_history_state],
        outputs=[chatbot, chat_history_state, user_input],
        queue=False
    ).then(
        fn=generate_assistant_reply,
        inputs=[pdf_content_state, chat_history_state],
        outputs=[chatbot, chat_history_state],
        queue=False
    )

    # Submit on Enter - Two-step callback
    user_input.submit(
        fn=add_user_message,
        inputs=[user_input, chat_history_state],
        outputs=[chatbot, chat_history_state, user_input],
        queue=False
    ).then(
        fn=generate_assistant_reply,
        inputs=[pdf_content_state, chat_history_state],
        outputs=[chatbot, chat_history_state],
        queue=False
    )

# Launch without specifying enable_queue, using default launch settings
if __name__ == "__main__":
    demo.launch()
