import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
import os

# Load environment variables
load_dotenv()

# Cache the document loading and processing
@st.cache_resource
def initialize_qa_chain():
    """Initialize the QA chain with document processing and model setup"""
    try:
        # Load and process documents
        loader = TextLoader("paul_graham_essay.txt")
        documents = loader.load()
        
        # Optimize chunk size and overlap for better context retention
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Reduced chunk size for more precise retrieval
            chunk_overlap=50,  # Reduced overlap to minimize redundancy
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],  # More granular splitting
            length_function=len
        )
        texts = text_splitter.split_documents(documents)
        
        # Initialize embeddings with error handling
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Create vector store
        vector_store = FAISS.from_documents(texts, embeddings)
        
        # Initialize LLM with optimized parameters
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            max_tokens=500,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            timeout=30  # Add timeout for API calls
        )
        
        # Create prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant that answers questions about the Paul Graham Essay.
                         Use the following context to answer the question at the end.
                         Be friendly and helpful in your responses. 
                         If you cannot answer the question from the context, just say "I don't know".
                         Always say "Thanks for asking!" at the end of the answer.
                         
                         Context: {context}"""),
            ("human", "{question}")
        ])
        
        # Set up RAG pipeline with search parameters
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(
                search_kwargs={"k": 3}  # Limit to top 3 most relevant chunks
            ),
            chain_type_kwargs={"prompt": prompt_template}
        )
        
        return qa_chain
    except Exception as e:
        st.error(f"Initialization Error: {str(e)}")
        return None

def main():
    # Set page config for better performance
    st.set_page_config(
        page_title="Paul Graham Essay Chatbot",
        page_icon="📚",
        layout="centered"
    )
    
    st.title("Paul Graham Essay Chatbot")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize QA chain
    qa_chain = initialize_qa_chain()
    if not qa_chain:
        st.error("Failed to initialize the chatbot. Please check your configuration.")
        return
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle user input
    query = st.chat_input("Ask me anything about the essay:")
    if query:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        # Generate AI response
        with st.spinner("Thinking..."):
            try:
                ai_response = qa_chain.run(query)
                
                # Add AI response to chat
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                with st.chat_message("assistant"):
                    st.markdown(ai_response)
                    
            except Exception as e:
                error_message = "Sorry, I encountered an error. Please try again."
                st.error(f"Error: {str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main()