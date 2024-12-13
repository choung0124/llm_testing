import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from mistralai import Mistral
from anthropic import Anthropic

from dotenv import load_dotenv
import os

load_dotenv()

st.session_state.model = ""
st.session_state.model_provider = ""
st.session_state.client = None
st.session_state.context = ""

with st.sidebar:
    model_provider = st.selectbox(
        "Choose a model provider", (
            "OpenAI",
            "Mistral",
            "Gemini",
            "Claude",
        ))
    
    # If a model provider is selected, allow the user to select a model
    provider_to_model = {
        "OpenAI": [
            "gpt-4o", "gpt-4o-mini",
            "o1-mini", "o1-preview"
            ],
        "Gemini": [
            "gemini-1.5-flash", "gemini-1.5-pro",
            "gemini-2.0-flash-exp", "gemini-exp-1206"
            ],
        "Mistral": ["mistral-large-latest"],
        "Claude": ["claude-3-5-sonnet-20240620"]
    }
    
    if model_provider:
        st.session_state.model_provider = model_provider
        st.session_state.model = st.selectbox(
            f"Choose a model for {model_provider}",
            provider_to_model[model_provider]
        )
    
    if model_provider and st.session_state.model:
        if model_provider == "OpenAI":
            st.session_state.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif model_provider == "Gemini":
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        elif model_provider == "Mistral":
            st.session_state.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        elif model_provider == "Claude":
            st.session_state.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

st.title("💬 Junwon Testing")

st.session_state.context = st.text_area("Context", value="Input your context here")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if st.session_state.context == "":
        st.error("Please input your context before asking a question")
        
    input_prompt = f"Context: {st.session_state.context}\nQuestion: {prompt}"
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    if st.session_state.model_provider == "OpenAI":
        messages = [{"role": "user", "content": input_prompt}]
        response = st.session_state.client.chat.completions.create(
            model=st.session_state.model,
            messages=messages,
        )
        st.chat_message("assistant").write(response.choices[0].message.content)
        st.session_state.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        
    elif st.session_state.model_provider == "Gemini":
        model = genai.GenerativeModel(st.session_state.model)
        response = model.generate_content(input_prompt)
        st.chat_message("assistant").write(response.text)
        st.session_state.messages.append({"role": "assistant", "content": response.text})
        
    elif st.session_state.model_provider == "Mistral":
        messages = [{"role": "user", "content": input_prompt}]
        response = st.session_state.client.chat.complete(
            model=st.session_state.model,
            messages=messages,
        )
        st.chat_message("assistant").write(response.choices[0].message.content)
        st.session_state.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        
    elif st.session_state.model_provider == "Claude":
        messages = [{"role": "user", "content": input_prompt}]
        response = st.session_state.client.messages.create(
            model=st.session_state.model,
            messages=messages,
            max_tokens=32768,
        )
        st.chat_message("assistant").write(response.content[0].text)
        st.session_state.messages.append({"role": "assistant", "content": response.content[0].text})
        