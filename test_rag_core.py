import pytest
from unittest.mock import patch, MagicMock
from rag_core import process_query

@patch('rag_core.retriever')
@patch('rag_core.groq_client')
def test_process_query(mock_groq_client, mock_retriever):
    # Mock the retriever's invoke method to return a list of mock documents
    mock_doc = MagicMock()
    mock_doc.page_content = "Theft is defined under Section 378 of the Pakistan Penal Code."
    mock_retriever.invoke.return_value = [mock_doc]

    # Mock the Groq client's response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "1. Case Type: Theft\n2. Section: 378"
    mock_groq_client.chat.completions.create.return_value = mock_response

    # Call the function
    query = "mera wallet cheen liya hai"
    result = process_query(query)

    # Assertions
    assert result["query"] == query
    assert "Theft" in result["answer"]
    assert "378" in result["answer"]
    assert result["context_found"] is True
    assert "Section 378" in result["context_used"]

    # Verify mocks were called
    mock_retriever.invoke.assert_called_once()
    mock_groq_client.chat.completions.create.assert_called_once()

@patch('rag_core.retriever')
@patch('rag_core.groq_client')
def test_process_query_no_context(mock_groq_client, mock_retriever):
    # Mock retriever returning empty
    mock_retriever.invoke.return_value = []

    # Call the function
    query = "asdfghjkl"
    result = process_query(query)

    # Assertions
    assert result["query"] == query
    assert result["context_found"] is False
    assert result["answer"] == "No relevant context found in the document."
    
    # Groq should not be called if no context is found
    mock_groq_client.chat.completions.create.assert_not_called()
