import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Load environment variables (e.g., Google API key)
load_dotenv()
st.set_page_config(
        page_title="Paul Graham Essay Chatbot",
        page_icon="📚",
        layout="centered",
        
    )
# Streamlit app title
st.title("Paul Graham Essay Chatbot")

# Initialize chat history in session state 
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load the .txt file containing the essay
loader = TextLoader("paul_graham_essay.txt")
documents = loader.load()

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Initialize Gemini embeddings 
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Create a FAISS vector store using Gemini embeddings 
vector_store = FAISS.from_documents(texts, embeddings)

# Initialize Gemini LLM 
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, max_tokens=500,request_options={"timeout": 5000})

#  Define a system prompt that uses "context"
system_prompt = """
You are an AI assistant that provides clear, insightful, 
and well-structured responses from the Paul Graham essay.
Be friendly and helpful in your responses.
Always ground responses in passages from Paul Graham’s essays.
Integrate them fluidly into responses without prefacing with “According to the text” or similar phrases
Do not attempt to generate speculative or unrelated content.
Get to the point efficiently without unnecessary framing.

Context: {context}
"""

# Create a prompt template with system and human messages, now including "context"
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    HumanMessagePromptTemplate.from_template("{question}")
])

#  Set up the RAG pipeline
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    chain_type_kwargs={"prompt": prompt_template}
)

# Chatbot interface elements
query = st.chat_input("Ask me anything about the essay:")

# Display chat history in the main area
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    # Add a spinner while the AI is generating a response
    with st.spinner("Chatbot is thinking..."):
        try:
            ai_response = qa_chain.run(query)
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            
            # Display the AI response in the chat
            with st.chat_message("assistant"):
                st.markdown(ai_response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error. Please try again."})
