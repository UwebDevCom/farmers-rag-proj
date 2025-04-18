import os
import openai
from transformers import GPT2Tokenizer



# Function to split text into chunks
def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Split the text into overlapping chunks.
    
    Args:
        text: The text to split.
        chunk_size: The size of each chunk in characters.
        overlap: The overlap between chunks in characters.
    
    Returns:
        List of text chunks.
    """
    # Handle empty or short texts
    if not text or len(text) <= chunk_size:
        return [text]
    
    # Split into chunks with overlap
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk:
            chunks.append(chunk)
    
    return chunks

def generate_embeddings(text):
    # Assuming you have an OpenAI API key set up
    openai.api_key = os.getenv("OPENAI_API_KEY")
      
      # Initialize the tokenizer
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # # Tokenize the input text
    # tokens = tokenizer.encode(text)
    # max_tokens = 8192  # Maximum tokens for the model

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Tokenize the input text
    tokens = tokenizer.encode(text)
    max_tokens = 8192  # Maximum tokens for the model
    
    embeddings = []
    
    # Process the text in chunks
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk)
        
        # Generate embeddings for each chunk
        response = openai.Embedding.create(input=chunk_text, model="text-embedding-ada-002")
        embedding = response['data'][0]['embedding']
        embeddings.append(embedding)

    # print(embeddings)
    pass
    return embedding
