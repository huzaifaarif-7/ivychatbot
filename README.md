# Ivy Chatbot

Ivy Chatbot is a Flask-based web application providing a chat interface powered by AI via OpenRouter. It includes features such as rate limiting, fallback responses, and Calendly integration for booking meetings.

## Prerequisites

- Python 3.8 or higher

## Installation

1. **Set up a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configuration:**
   Ensure the `.env` file is present in the root of the project with required configurations, such as your `OPENROUTER_API_KEY` and `SECRET_KEY`.

## Running the Application

To start the development server, run:

```bash
python app.py
```

The application will run locally, and you can access the chat interface at:
`http://127.0.0.1:5000`

## Features

- **OpenRouter AI Integration:** Communicates with LLMs using OpenRouter.
- **Fallback Mechanism:** Uses a static knowledge base when the AI service is unavailable.
- **Calendly Support:** Detects meeting requests and offers a scheduling link.
- **Rate Limiting:** Protects endpoints from abuse.
