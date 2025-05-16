# RAG-ZhijieWorks

## Introduction
This is a simple Retrieval Augmented Generation (RAG) app to ask question to my academic publications. 
These publications are PDF files in the **data** folder. 
The text is extracted by the **PyPDF2**. 

This app is developed by **langchain** framework.
The **Gemini** model from google has been used for embedding and chat, but it can be switched to other models easily. 

## Usage
In command line, run
```
python pdf-rag.py
```
And then input your prompt. 

## TODO
Only the text of the PDF files is extracted. 
Some other methods like **Unstructured** can be tried.
