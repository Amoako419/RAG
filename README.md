# RAG Application on Paul Graham Essay

## Overview
This application is designed to analyze and generate insights from essays written by Paul Graham using Retrieval-Augmented Generation (RAG) techniques.

## Features
- **Text Retrieval**: Efficiently retrieve relevant sections from Paul Graham's essays.
- **Text Generation**: Generate new text based on the retrieved sections.
- **Analysis Tools**: Tools to analyze the content and structure of the essays.

## Installation
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage
1. **Data Preparation**: Ensure that the essay is available in the `paul_graham_essay.txt` file.
2. **Run the Application**: Execute the main script to start the analysis.
```bash
python main.py
```
3. **Streamlit Interface**: Launch the Streamlit interface for interactive analysis.
```bash
streamlit run app.py
```

## Directory Structure
```
RAG/
├── paul_graham_essay.txt      # File containing Paul Graham's essay
├── src/                       # Source code for the application
├── tests/                     # Unit tests for the application
├── app.py                     # Streamlit application file
├── README.md                  # This README file
└── requirements.txt           # List of dependencies
```

## Technology Stack
- **LangChain**: For building the retrieval and generation pipelines.
- **Gemini 2.0 Flash**: For efficient and scalable text processing.
- **Streamlit**: For creating an interactive web interface.

## Streamlit Link
Access the Streamlit application [here](https://paul-essays.streamlit.app/).

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

## License
This project is licensed under the MIT License.
