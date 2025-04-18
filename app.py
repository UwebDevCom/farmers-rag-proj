import streamlit as st
from embedding import generate_embeddings, chunk_text
from vec_store import store_embedding, query_embedding
from read_files import read_file
import tempfile
import os
import openai
import hashlib
import uuid
import textwrap




if 'page' not in st.session_state:
    st.session_state.page = 'home'


# Sidebar navigation
st.sidebar.title("Navigation")

# Navigation buttons
if st.sidebar.button("Home"):
    st.session_state.page = 'home'
if st.sidebar.button("Search Documents"):
    st.session_state.page = 'chat'


def home_page():
    st.title('AI Chat with File Embedding') 
    st.write('This is a chatbot that can answer questions about a document.')
    
    # Ensure the 'files' directory exists
    os.makedirs('files', exist_ok=True)
    # File uploader
    uploaded_file = st.file_uploader("Choose a file")

    
    if uploaded_file is not None:
        # Save the uploaded file to the 'files' directory
        file_path = os.path.join('files', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.read())

        # Read the file content using read_file function
        file_content = read_file(file_path)
        print(file_path)

        # Split the content into chunks
        chunks = chunk_text(file_content)
        
        # Create a base ASCII-compatible ID using a hash of the filename
        base_id = hashlib.md5(uploaded_file.name.encode('utf-8')).hexdigest()
        
        # Process each chunk
        with st.status("Processing document chunks..."):
            for i, chunk in enumerate(chunks):
                # Generate embeddings for the chunk
                chunk_embeddings = generate_embeddings(chunk)
                
                # Create a unique ID for the chunk
                chunk_id = f"{base_id}-chunk-{i}"
                
                # Store embeddings in Pinecone with chunk metadata
                metadata = {
                    'id': chunk_id,
                    'original_filename': uploaded_file.name,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_text': chunk[:200] + "..." if len(chunk) > 200 else chunk  # Store preview of chunk text
                }
                store_embedding(chunk_embeddings, metadata)
                st.write(f"Processed chunk {i+1}/{len(chunks)}")
        
        st.success(f'File embedded successfully! Split into {len(chunks)} chunks.')




def upload_file():
    st.title('Upload a file')

def chat_page():
    st.title('Chat with the document')

# Function to generate a response using a language model
    def generate_response(query, context):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Create prompt with context from chunks
        prompt = f"""
        Context information from document chunks:
        {context}

        Question: {query}
        """
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides accurate information based on the context provided."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content.strip()


    # AI chat functionality
    query = st.text_input(key="query", label="Enter your question here")
    
    if query:
        # Generate embeddings for the query
        query_embeddings = generate_embeddings(query)
        
        # Retrieve similar chunks from Pinecone
        similar_chunks = query_embedding(query_embeddings)
        print(similar_chunks)

        if similar_chunks and hasattr(similar_chunks, 'matches'):
            # Display retrieved chunks in expander for transparency
            with st.expander("Retrieved document chunks"):
                for i, match in enumerate(similar_chunks.matches[:3]):  # Show top 3 matches
                    st.markdown(f"**Match {i+1}** (Score: {match.score:.4f})")
                    print(match)
                    if hasattr(match, 'metadata') and match.metadata and match.metadata.get('chunk_text'):
                        st.text(match.metadata.get('chunk_text'))
                    st.divider()
            
            # Extract context from the chunk metadata
            context_chunks = []
            
            for match in similar_chunks.matches:
                if hasattr(match, 'metadata') and match.metadata and match.metadata.get('chunk_text'):
                    context_chunks.append(match.metadata.get('chunk_text'))
            
            # Combine all relevant chunks into one context
            context = "\n\n---\n\n".join(context_chunks)
            
            # Generate a response using the language model
            with st.status("Generating answer..."):
                answer = generate_response(query, context)
            
            # Display the answer
            st.markdown("### Answer")
            st.markdown(answer)

            print(answer)
        else:
            st.error("No relevant information found. Try a different question or upload more documents.")


# Display the selected page
print(st.session_state)

if st.session_state.get('page') == 'home':
    home_page()

elif st.session_state.get('page') == 'chat':
    chat_page()

