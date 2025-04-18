import unittest
from unittest.mock import patch
from app import generate_response, query_embedding

class TestChatFunctionality(unittest.TestCase):
    @patch('app.generate_embeddings')
    @patch('app.query_embedding')
    @patch('app.generate_response')
    def test_chat_input(self, mock_generate_response, mock_query_embedding, mock_generate_embeddings):
        # Mock the generate_embeddings function
        mock_generate_embeddings.return_value = ['mocked_vector']

        # Mock the query_embedding function
        mock_query_embedding.return_value = ['mocked_embedding']

        # Mock the generate_response function
        mock_generate_response.return_value = 'This is a mocked response.'

        # Simulate user input
        query = 'What is the capital of France?'
        
        # Call the mock functions directly
        embedding_vector = mock_generate_embeddings(query)
        similar_embeddings = mock_query_embedding(embedding_vector)
        context = "\n".join([str(item) for item in similar_embeddings])
        answer = mock_generate_response(query, context)

        # Assert the response
        self.assertEqual(answer, 'This is a mocked response.')

if __name__ == '__main__':
    unittest.main() 