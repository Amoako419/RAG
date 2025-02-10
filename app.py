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

# Streamlit app title
st.title("AmaliAI Chatbot")

# Initialize chat history in session state if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Step 1: Load the .txt file (moved to the top for clarity)
loader = TextLoader("paul_graham_essay.txt")
documents = loader.load()

# Step 2: Split the text into chunks (moved to the top for clarity)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Step 3: Initialize Gemini embeddings (moved to the top for clarity)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Step 4: Create a FAISS vector store using Gemini embeddings (moved to the top for clarity)
vector_store = FAISS.from_documents(texts, embeddings)

# Step 5: Initialize Gemini LLM (moved to the top for clarity)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, max_tokens=500)

# Step 6: Define a system prompt that uses "context"
system_prompt = """
You are an AI assistant that answers questions based on the provided document context.
Use the following pieces of context to answer the question at the end.
If you cannot answer the question from the context, just say "I don't know", don't try to make up an answer.

Context: {context}
"""

# Create a prompt template with system and human messages, now including "context"
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

# Chatbot interface elements
query = st.chat_input("Ask me anything about the essay:")

if query:
    st.session_state.chat_history.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        ai_response = qa_chain.run(query)
        full_response = ai_response

        message_placeholder.markdown(full_response)

    st.session_state.chat_history.append({"role": "assistant", "content": full_response})

# # **UI Improvement: Add a separator and title for chat history**
# st.markdown("---") # Separator line
# st.subheader("Conversation History") # Title for the history section

# Display chat history in the main area
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])