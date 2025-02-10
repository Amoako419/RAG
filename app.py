import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Load environment variables (e.g., Google API key)
load_dotenv()

# Streamlit app title
st.write("AmaliAI")

# Step 1: Load the .txt file
loader = TextLoader("paul_graham_essay.txt")
documents = loader.load()

# Step 2: Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Step 3: Initialize Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Step 4: Create a FAISS vector store using Gemini embeddings
vector_store = FAISS.from_documents(texts, embeddings)

# Step 5: Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=500)

# Step 6: Define a system prompt
system_prompt = """
You are an AI assistant that answers questions based on the provided document. 
Your answers should be concise, accurate, and directly relevant to the document's content.
If you don't know the answer, say "I don't know" instead of making up an answer.
"""

# Create a prompt template with system and human messages
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    HumanMessagePromptTemplate.from_template("{question}")
])

# Step 7: Set up the RAG pipeline
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    chain_type_kwargs={"prompt": prompt_template}
)

# Step 8: Query the RAG pipeline
query = st.text_input("Ask a question about the document:")
if query:
    response = qa_chain.run(query)
    st.write("Answer:")
    st.write(response)