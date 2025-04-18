from pydantic_ai import Agent
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import openai


from embedding import generate_embeddings
from vec_store import store_embedding, query_embedding
from read_files import read_file
load_dotenv()


class EmbeddingAgent(Agent):
    def __init__(self):
        super().__init__(model="claude-3-5-sonnet-20240620")
    
    def process_file(self, file_path):
        text = read_file(file_path)
        embedding = generate_embeddings(text)
        metadata = {'id': os.path.basename(file_path)}
        store_embedding(embedding, metadata)

    def answer_question(self, question):
        response = self.run_sync(question)
        return response
    

agent = EmbeddingAgent()

# Get all files from the 'files' directory
files_directory = 'files'
if os.path.exists(files_directory) and os.path.isdir(files_directory):
    for filename in os.listdir(files_directory):
        file_path = os.path.join(files_directory, filename)
        print(file_path)
        if os.path.isfile(file_path):
            print(f"Processing file: {filename}")
            agent.process_file(file_path)
else:
    print(f"Directory '{files_directory}' not found or is not a directory")

answer_question = agent.answer_question("What is the main topic of the document?")
print(answer_question)

def get_prompt_embedding(prompt):
       response = openai.Embedding.create(input=prompt, model="text-embedding-ada-002")
       return response['data'][0]['embedding']

prompt = "מה זה בלק דניאמוד?"
embeded_query = get_prompt_embedding(prompt)
prompt = query_embedding(embeded_query)
print(prompt)
# agent = Agent(
#     model="claude-3-5-sonnet-20240620",
#     system_prompt="You are a helpful assistant that can answer questions and help with tasks."
# )

# human = input("Enter a question: ")
# response = agent.run_sync(human)


# print(response)
