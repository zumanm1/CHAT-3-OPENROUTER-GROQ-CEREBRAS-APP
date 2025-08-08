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
3. Create a \`.env\` file with your API keys:
   \`\`\`
   OPENROUTER_API_KEY=your_openrouter_api_key
   GROQ_API_KEY=your_groq_api_key
   CEREBRAS_API_KEY=your_cerebras_api_key
   \`\`\`
4. Run the app: \`streamlit run app.py\`

## Usage
Select your preferred provider and model from the sidebar, then start chatting in the main interface.

Note: Models may have usage limits depending on your account tier with each provider.

