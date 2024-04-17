# Fun Jarvis v2

Fun Jarvis v2 is an advanced text processing and information retrieval system designed to answer complex queries by analyzing textual data from various sources like books, articles, and documents. It utilizes state-of-the-art natural language processing (NLP) models and techniques to perform tasks such as embedding generation, context retrieval, and re-ranking of text passages to generate accurate and context-aware responses.

## Features

- **Deep Query Understanding**: Decomposes complex queries into simpler sub-questions that are easier to answer.
- **Textual Embeddings**: Utilizes pre-trained models for generating textual embeddings that capture the semantic meaning of the text.
- **Iterative Context Building**: Gathers relevant information iteratively to build context and improve the accuracy of answers.
- **Dynamic Text Reranking**: Employs reranking algorithms to prioritize text passages based on relevance to the query.
- **Information Extraction**: Implements advanced extraction mechanisms to pull information from structured and unstructured data.

## Installation

To set up the project, follow these steps:

1. Clone the repository:

   ```
   git clone https://github.com/rohanawhad/fun_jarvis_v2.git
   ```

2. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables or configuration files as needed for API keys and model paths.

## Usage

### Running the System

To run Fun Jarvis v2:

```
python main.py
```

This will start the system and load the necessary models and configurations as defined in `config.py`.

### Query Processing

Edit the `question.txt` file to input your query, or modify the `main.py` script to accept dynamic queries from the command line or another interface.

### Text Embedding and Retrieval

The system will automatically handle embedding generation and retrieval of text passages from the provided sources using the models and techniques specified in the `encoders` directory.

## Development

### Modules

- **Main Module (`main.py`)**: Orchestrates the query processing, context retrieval, and response generation.
- **Configuration (`config.py`)**: Manages configuration variables like model paths and operational parameters.
- **Encoders (`encoders/`)**: Contains different encoder modules for generating textual embeddings.
- **Helper Functions (`helper.py`)**: Provides utility functions for data management and interaction.
- **Prompts (`prompts.py`)**: Manages the prompt templates used for interacting with language models.

### Adding New Features

1. Create a new branch for your feature.
2. Implement your feature with corresponding unit tests.
3. Submit a pull request to the main branch.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.
