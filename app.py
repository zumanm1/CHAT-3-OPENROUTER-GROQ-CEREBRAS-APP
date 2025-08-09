import streamlit as st
import os
import requests
import json
from dotenv import load_dotenv

# Set page config
st.set_page_config(page_title="Multi-Provider LLM Chat", layout="wide")

# Load environment variables
load_dotenv()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Predefined model lists for each provider
LLM_MODELS = {
    "openrouter": [
        "meta-llama/llama-4-maverick:free",
        "meta-llama/llama-4-scout:free",
        "deepseek/deepseek-v3-base:free",
        "deepseek/deepseek-chat-v3-0324:free",
        "deepseek/deepseek-r1-zero:free",
        "nvidia/llama-3.1-nemotron-nano-8b-v1:free",
        "mistralai/mistral-small-3.1-24b-instruct:free",
        "nousresearch/deephermes-3-llama-3-8b-preview:free",
        "qwen/qwen2.5-vl-3b-instruct:free",
        "moonshotai/kimi-vl-a3b-thinking:free"
    ],
    "groq": [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "gemma2-9b-it",
        "meta-llama/llama-guard-4-12b",
        "whisper-large-v3",
        "whisper-large-v3-turbo",
        "deepseek-r1-distill-llama-70b",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-prompt-guard-2-22m",
        "meta-llama/llama-prompt-guard-2-86m",
        "moonshotai/kimi-k2-instruct",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3-32b",
        "playai-tts",
        "playai-tts-arabic",
        "compound-beta",
        "compound-beta-mini"
    ],
    "cerebras": [
        "llama3.1-8b",
        "llama-3.3-70b",
        "llama-4-scout-17b-16e-instruct",
        "qwen-3-32b",
        "qwen-3-235b-a22b-instruct-2507",
        "qwen-3-235b-a22b-thinking-2507",
        "qwen-3-coder-480b",
        "gpt-oss-120b",
        "deepseek-r1-distill-llama-70b",
        "cerebras-gpt-13b",
        "cerebras-gpt-6.7b",
        "cerebras-gpt-2.7b",
        "cerebras-gpt-1.3b",
        "cerebras-gpt-590m",
        "cerebras-gpt-256m",
        "cerebras-gpt-111m",
        "btlm-3b-8k-base",
        "gigaGPT-111b"
    ],
    "ollama": [
        "llama3",
        "llama3.1",
        "llama3.2",
        "llama3.3",
        "mistral",
        "mixtral",
        "gemma",
        "gemma2",
        "phi3",
        "qwen",
        "qwen2",
        "llama-guard-4"
    ]
}

class LLMClient:
    def __init__(self, provider):
        self.provider = provider.lower()
        self.api_key = None
        self.base_url = None
        self.headers = {}
        
        # Load API key based on provider
        if self.provider == "openrouter":
            self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip().split('#')[0].strip().strip('"\'')
            self.base_url = "https://openrouter.ai/api/v1"
            if not self.api_key:
                st.error("🚨 OpenRouter API key not found! Please add your API key to the .env file.")
                st.stop()
            if not self.api_key.startswith("sk-or-v1-"):
                st.error("🚨 Invalid OpenRouter API key format! Your API key must start with 'sk-or-v1-'")
                st.error("Please verify your API key at: https://openrouter.ai/account")
                st.stop()
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            self.app_url = os.getenv("OPENROUTER_APP_URL", "").split('#')[0].strip()
            self.app_name = os.getenv("OPENROUTER_APP_NAME", "Streamlit Chat App").split('#')[0].strip()
            if self.app_url:
                self.headers["HTTP-Referer"] = self.app_url
            if self.app_name:
                self.headers["X-Title"] = self.app_name
                
        elif self.provider == "groq":
            self.api_key = os.getenv("GROQ_API_KEY", "").strip().split('#')[0].strip().strip('"\'')
            self.base_url = "https://api.groq.com/openai/v1"
            if not self.api_key:
                st.error("🚨 Groq API key not found! Please add your API key to the .env file.")
                st.stop()
            if not self.api_key.startswith("gsk_"):
                st.error("🚨 Invalid Groq API key format! Your API key must start with 'gsk_'")
                st.stop()
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
        elif self.provider == "cerebras":
            self.api_key = os.getenv("CEREBRAS_API_KEY", "").strip().split('#')[0].strip().strip('"\'')
            self.base_url = "https://api.cerebras.ai/v1"
            if not self.api_key:
                st.error("🚨 Cerebras API key not found! Please add your API key to the .env file.")
                st.stop()
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
        elif self.provider == "ollama":
            # Ollama typically runs locally, doesn't require API key
            self.base_url = "http://localhost:11434/api"
            self.headers = {
                "Content-Type": "application/json",
            }
            
            # Check if Ollama is running
            try:
                ollama_check = requests.get("http://localhost:11434/api/tags", timeout=5)
                if ollama_check.status_code != 200:
                    st.error("🚨 Ollama is not running locally! Please start Ollama first.")
                    st.stop()
            except requests.exceptions.ConnectionError:
                st.error("🚨 Ollama is not running locally! Please start Ollama first.")
                st.stop()
            except Exception as e:
                st.error(f"🚨 Error connecting to Ollama: {str(e)}")
                st.stop()
    
    def list_models(self):
        """Return predefined model list for the provider."""
        return LLM_MODELS.get(self.provider, [])
    
    def chat(self, model, messages):
        """Send chat request to the selected API provider."""
        # Use the exact payload format
        payload = {
            "model": model,
            "messages": messages
        }
        
        # Add provider-specific headers
        headers = self.headers.copy()
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = os.getenv("OPENROUTER_APP_URL", "http://localhost:8501")
            headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME", "Streamlit Chat App")
        
        # Determine the correct endpoint
        if self.provider == "ollama":
            url = f"{self.base_url}/chat"
        else:
            url = f"{self.base_url}/chat/completions"
        
        try:
            response = requests.post(
                url=url,
                headers=headers,
                data=json.dumps(payload),
                timeout=60
            )
            
            # Check response status
            if response.status_code == 200:
                response_data = response.json()
                if self.provider == "ollama":
                    # Ollama response format
                    if "message" in response_data:
                        return response_data["message"]["content"]
                    else:
                        st.error("🚨 Unexpected response format from Ollama API")
                        return None
                else:
                    # Other providers response format
                    if "choices" in response_data and len(response_data["choices"]) > 0:
                        return response_data["choices"][0]["message"]["content"]
                    else:
                        st.error("🚨 Unexpected response format from API")
                        return None
            
            elif response.status_code == 401:
                st.error("🚨 Authentication failed!")
                st.error(f"Trying alternative format for {self.provider}...")
                
                # Try without additional headers as fallback
                minimal_headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                fallback_response = requests.post(
                    url=url,
                    headers=minimal_headers,
                    data=json.dumps(payload),
                    timeout=60
                )
                
                if fallback_response.status_code == 200:
                    st.success("✅ Fallback method worked!")
                    response_data = fallback_response.json()
                    if self.provider == "ollama":
                        return response_data["message"]["content"]
                    else:
                        return response_data["choices"][0]["message"]["content"]
                else:
                    st.error(f"🚨 Fallback also failed: {fallback_response.text}")
                    return None
                
            elif response.status_code == 400:
                st.error("🚨 Bad Request - Invalid parameters")
                st.error(f"Response: {response.text}")
                return None
                
            elif response.status_code == 429:
                st.error("🚨 Rate limited - Please wait and try again")
                return None
                
            else:
                st.error(f"🚨 API Error {response.status_code}")
                st.error(f"Response: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            st.error("🚨 Request timed out - Please try again")
            return None
        except requests.exceptions.ConnectionError:
            st.error("🚨 Connection error - Please check your internet connection")
            return None
        except Exception as e:
            st.error(f"🚨 Unexpected error: {str(e)}")
            return None

# Provider selection
with st.sidebar:
    st.header("🤖 Provider & Model Selection")
    
    # Provider selection dropdown
    provider = st.selectbox(
        "Choose an LLM provider:",
        options=["OpenRouter", "Groq", "Cerebras", "Ollama"],
        index=0,
        help="Select the LLM provider you want to use"
    )
    
    # Add info about the selected provider
    if provider == "OpenRouter":
        st.info("Using OpenRouter API to access various AI models")
    elif provider == "Groq":
        st.info("Using Groq API for fast inference with Llama models")
    elif provider == "Cerebras":
        st.info("Using Cerebras API for Llama and Qwen models")
    elif provider == "Ollama":
        st.info("Using Ollama for locally hosted AI models")
    
    # Create client instance for selected provider
    try:
        client = LLMClient(provider)
    except Exception as e:
        st.error(f"Failed to initialize {provider} client: {str(e)}")
        st.stop()
    
    # Get predefined models for the selected provider
    models = client.list_models()
    
    if models:
        selected_model = st.selectbox(
            "Choose a model:",
            options=models,
            index=0,
            help=f"Select from available models on {provider}"
        )
        
        # Show selected model info
        st.success(f"✅ Using: `{selected_model}`")
        
        # Add model info if available
        if "gemma" in selected_model.lower():
            st.caption("🧠 Google's Gemma model - Great for general chat")
        elif "llama" in selected_model.lower():
            st.caption("🦙 Meta's LLaMA model - Excellent for reasoning")
        elif "phi" in selected_model.lower():
            st.caption("📱 Microsoft's Phi model - Efficient and fast")
        elif "qwen" in selected_model.lower():
            st.caption("🚀 Alibaba's Qwen model - Strong multilingual capabilities")
        elif "deepseek" in selected_model.lower():
            st.caption("💻 DeepSeek model - Powerful coding and reasoning")
        elif "mistral" in selected_model.lower():
            st.caption("🔥 Mistral model - Efficient and versatile")
        elif "nousresearch" in selected_model.lower():
            st.caption("🔬 Nous Research model - Experimental and advanced")
        elif "nvidia" in selected_model.lower():
            st.caption("🎨 NVIDIA model - Specialized in various tasks")
        elif "moonshotai" in selected_model.lower():
            st.caption("🌙 Moonshot AI model - Innovative and capable")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", help="Clear all messages"):
            st.session_state.messages = []
            st.rerun()
            
    else:
        st.error(f"❌ No models available for {provider}")
        st.error("Please check your API key and internet connection")
        st.stop()

# Main chat interface
st.title("💬 Multi-Provider LLM Chat")
st.caption(f"Chatting with **{selected_model}** via {provider} API")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?", key="chat_input"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = client.chat(selected_model, st.session_state.messages)
            
            if response:
                st.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.error("Failed to get response. Please try again.")

# Footer
st.divider()
st.caption(f"Powered by {provider} API • Models may have usage limits")
