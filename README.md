# CHAT-3-OPENROUTER-GROQ-CEREBRAS

A Streamlit chat application that supports multiple LLM providers:
- OpenRouter
- Groq
- Cerebras

## Features
- Switch between different LLM providers
- Select from predefined models for each provider
- Chat interface with message history
- Error handling for API issues

## Setup
1. Clone this repository
2. Install requirements: \`pip install -r requirements.txt\`
3. Copy the \`.env.example\` file to \`.env\` and add your API keys:
   \`\`\`
   cp .env.example .env
   \`\`\`
   Then edit the .env file to add your actual API keys
4. Run the app: \`streamlit run app.py\`

## Usage
Select your preferred provider and model from the sidebar, then start chatting in the main interface.

Note: Models may have usage limits depending on your account tier with each provider.
