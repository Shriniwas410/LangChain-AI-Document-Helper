# LangChain-AI-Document-Helper

This Streamlit application leverages LangChain AI tools to provide an interactive chat interface that helps users query and retrieve information from uploaded PDF documents. The application supports two models: Gemini and OpenAI, allowing users to choose based on their preference or specific needs.

## Features

- **Document Upload**: Users can upload PDF documents which the application processes to answer queries.
- **Model Selection**: Users can select between Gemini and OpenAI models.
- **Real-Time Interaction**: The application provides real-time responses to user queries using the selected AI model.

## Getting Started

### Prerequisites

- Python 3.6+
- Streamlit
- LangChain and its dependencies

### Installation

1. Clone the repository:
```git clone https://github.com/Shriniwas410/LangChain-AI-Document-Helper.git```

2. Navigate to the project directory:
```cd LangChain-AI-Document-Helper```

3. Install the required packages:
```pip install -r requirements.txt```

### API Keys

#### Gemini API Key

To use the Gemini model, you need an API key from Google Cloud Platform:

1. Visit the [Google AI Studioio](https://aistudio.google.com/).
2. Create a new API key.
3. Click on "Create Credentials" and select "API key".
4. Restrict the API key as necessary for security.

Store your API key securely and use it in the application by entering it in the sidebar when prompted.

#### OpenAI API Key

To use the OpenAI model, you need an API key from OpenAI:

1. Visit [OpenAI's API platform](https://platform.openai.com/signup).
2. Sign up or log in to create an API key.
3. Navigate to the API keys section and generate a new key.

Store your API key securely and use it in the application by entering it in the sidebar when prompted.

### Running the Application

Run the application using Streamlit:
```streamlit run Document_helper.py ```

The application will be available in your web browser at `http://localhost:8501`.

## Usage

1. Select the AI model from the sidebar.
2. Enter the respective API key for the selected model.
3. Upload PDF documents via the sidebar uploader.
4. Use the chat interface to ask questions and receive information based on the uploaded documents.
