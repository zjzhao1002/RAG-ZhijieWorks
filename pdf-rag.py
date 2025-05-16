import os
import getpass
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

file_path = "./data/"
dir_list = os.listdir(file_path)

pdf_files = []
for f in dir_list:
    if "pdf" in f:
        pdf_files.append(os.path.join(file_path, f))

text = ""
for pdf in pdf_files:
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
chunks = text_splitter.split_text(text=text)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
vector_store.save_local("faiss_index")

prompt_template = """
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you do not know the answer, just say "answer is not available in this context". 
Do not provide wrong answer.
Use three sentences maximum and keep the answer concise.\n
Context: {context}\n
Question: {question}\n
Answer: 
"""

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain = prompt | model

while True:
    print("\n\n----------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n")
    if question == "q":
        break

    context = vector_store.similarity_search(question)

    response = chain.invoke({"context": context, "question": question})
    print(response.content)