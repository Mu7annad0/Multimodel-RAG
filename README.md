# Multimodel RAG

## Overview
Multimodal RAG system that ingests data containing images, text, and tables from the well known paper “Attention Is All You Need”. The system able to retrieve relevant information and reason over the multimodal data to answer user queries.

## Installation
To install the necessary dependencies, you can use:

#### 1. Add openai key
```sh
export OPENAI_API_KEY=<ADD YOUR API KEY HERE>
```
#### 2. Add llama cloud key for the llamaparse
```sh
export LLAMA_CLOUD_API_KEY=<ADD YOUR API CLOUD API KEY HERE>
```

####  3. Install the dependencies
```sh
pip install -r requirements.txt
```
#### 4. Run the Application
```sh
streamlit run app.py
```