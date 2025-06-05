from rag.embeddings import get_vector_store
from rag.rag_chain import query_rag

def classify(result_path="results/classification/"):
    """Classify product reviews into topics and write output to file"""
    question = """
    Please extract the topics mentioned in the e-commerce product reviews provided. A few topic examples are: product quality, 
    delivery, customer service, packaging and usage. For each topic, add how many reviews belong to it. 
    A review can belong to multiple topics. Present the result in a table where each row has a pair of topics and review count, 
    write the result in portuguese.
    """
    query_rag(question, get_vector_store(), "classification", result_path)

if __name__ == "__main__":
    # TODO: In case the index doesn't exist, create it by calling
    #  index_documents(load_data()) instead of get_vector_store()
    classify()

