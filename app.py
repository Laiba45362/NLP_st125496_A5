import os
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

# Define the model name and directory
model_name = "gpt2"
model_dir = "reward_model"

# Create the directory if it doesn't exist
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Function to check if all required files exist
def check_files():
    required_files = [
        "config.json", "pytorch_model.bin", "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json", "vocab.json"
    ]
    return all([os.path.exists(os.path.join(model_dir, file)) for file in required_files])

# Download files if they don't exist
if not check_files():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    print("Downloading model files...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print("Download complete.")

# ✅ Set device (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load reward model (For scoring)
reward_model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=1).to(device)
state_dict = torch.load("reward_model.pth", map_location=device)
reward_model.load_state_dict(state_dict, strict=False)
reward_model.eval()

# ✅ Load text generation model (For generating responses)
response_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
response_tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ Load tokenizer and fix padding issue
reward_tokenizer = AutoTokenizer.from_pretrained(model_dir)
if reward_tokenizer.pad_token is None:
    reward_tokenizer.pad_token = reward_tokenizer.eos_token  # Set EOS token as padding

# 🎯 Function to generate AI response + Score
def generate_response(prompt):
    input_ids = response_tokenizer.encode(prompt, return_tensors="pt").to(device)
    response_output = response_model.generate(input_ids, max_length=100, num_return_sequences=1)
    response_text = response_tokenizer.decode(response_output[0], skip_special_tokens=True)
    
    inputs = reward_tokenizer(response_text, truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        output = reward_model(**inputs)
    
    score = output.logits.item()
    probability = torch.sigmoid(torch.tensor(score)).item()
    return response_text, probability

# 🌟 Modern UI Design
st.set_page_config(page_title="AI Response Evaluator", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
        body { background-color: #f4f4f4; }
        .stApp { background: white; padding: 2rem; border-radius: 12px; box-shadow: 0px 4px 10px rgba(0,0,0,0.1); }
        h1 { color: #007bff; text-align: center; font-size: 2.8rem; }
        .stTextInput>div>div>textarea { font-size: 1.2rem; padding: 10px; border-radius: 8px; border: 1px solid #007bff; }
        .stButton>button { background: linear-gradient(135deg, #007bff, #00d4ff); color: white; font-size: 1.2rem; border-radius: 8px; padding: 12px 24px; border: none; }
        .stButton>button:hover { background: linear-gradient(135deg, #0056b3, #0099cc); }
        .stSuccess, .stInfo { font-size: 1.2rem; padding: 15px; border-radius: 8px; border: 1px solid #007bff; }
        .footer { text-align: center; font-size: 1rem; color: #6c757d; padding-top: 20px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1>🤖 AI Response Evaluator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:1.3rem; color:#007bff;'>Evaluate AI-generated responses with confidence scores!</p>", unsafe_allow_html=True)

user_input = st.text_area("💬 Enter a prompt:", "What is the capital of France?", height=150)

if st.button("🔍 Generate Response"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a valid prompt!")
    else:
        ai_response, probability = generate_response(user_input)
        st.success(f"**📝 AI Response:** {ai_response}")
        st.info(f"**📊 Confidence Score: {probability:.4f}** (Higher is better)")
        
        if probability > 0.8:
            st.markdown("✅ This response is highly relevant!", unsafe_allow_html=True)
        elif probability > 0.5:
            st.markdown("⚖️ This response is moderately relevant.", unsafe_allow_html=True)
        else:
            st.markdown("❌ This response is less relevant.", unsafe_allow_html=True)

st.markdown("<p class='footer'>✨ Built with Streamlit & PyTorch | Created by St125050</p>", unsafe_allow_html=True)
