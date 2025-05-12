from rag.embeddings import get_vector_store
from rag.rag_chain import query_rag

if __name__ == "__main__":
    # TODO: In case the index doesn't exist, create it by calling
    #  index_documents(load_data()) instead of get_vector_store()
    question = "Please extract the topics mentioned in the review texts."
    result = query_rag(question, get_vector_store())
    print(result)