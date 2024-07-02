# AI Book Conversation Manager

AI Book Conversation Manager is a Flask-based web application that allows users to upload books, process them using AI, and have intelligent conversations about the content. This application leverages the power of language models to provide insightful responses based on the uploaded books.

## Features

- Upload and manage EPUB books
- Process books using AI (Google Generative AI)
- Start new conversations about books
- Continue existing conversations
- Delete individual conversations or entire books
- User-friendly interface with modern CSS styling

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7+
- pip (Python package manager)
- A Google API key for accessing Google Generative AI services

## Installation

1. Clone the repository
2. Create a virtual environment (optional but recommended)
3. Install the required packages

4. Set up your Google API key:
- Obtain a Google API key from the Google Cloud Console
- Set it as an environment variable:
  ```
  export GOOGLE_API_KEY=your_api_key_here
  ```
  On Windows, use `set` instead of `export`

## Usage

1. Run the Flask application 

2. Open a web browser and navigate to `http://localhost:5000`

3. Upload a book (EPUB format) using the "Upload Book" button

4. Click on "Conversations" next to a book to start or continue conversations

5. Enter your questions or prompts in the text area and click "Send" to get AI-generated responses

6. Manage your conversations and books using the provided interface

## Contributing

Contributions to the AI Book Conversation Manager are welcome. Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

- [Flask](https://flask.palletsprojects.com/)
- [LangChain](https://python.langchain.com/)
- [Google Generative AI](https://cloud.google.com/ai-platform/prediction/docs/overview)
- [FAISS](https://github.com/facebookresearch/faiss)

## Contact

If you have any questions or feedback, please open an issue on the GitHub repository.