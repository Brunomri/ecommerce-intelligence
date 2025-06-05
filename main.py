from rag.embeddings import get_vector_store
from rag.rag_chain import query_rag

def classify(result_path="results/classification/"):
    """Classify product reviews into topics and write output to file"""
    question = """
    Please extract the topics mentioned in the e-commerce product reviews provided. A few topic examples are: product quality, 
    delivery, customer service, packaging and usage. For each topic, add how many reviews belong to it. 
    A review can belong to multiple topics. Present the result in a table where each row has the topic and its review count, 
    write the result in portuguese.
    """
    query_rag(question, get_vector_store(), "classification", result_path)

def summarize_by_product(result_path="results/summarization/product/"):
    """Summarize reviews by product and write output to file"""
    question = """
    Please summarize the reviews by field product_id. 
    Present the result in a table where each row has the product ID, product name and summary, write the result in portuguese.
    Ignore reviews where the field product_name is blank.
    """
    query_rag(question, get_vector_store(), "summary_by_product", result_path)

def summarize_by_brand(result_path="results/summarization/brand/"):
    """Summarize reviews by brand and write output to file"""
    question = """
    Please summarize the reviews of all products from a certain brand using field product_brand.
    Present the result in a table where each row has the product brand and summary, write the result in portuguese.
    Ignore reviews where the field product_brand is blank.
    """
    query_rag(question, get_vector_store(), "summary_by_brand", result_path)

if __name__ == "__main__":
    # TODO: In case the index doesn't exist, create it by calling
    #  index_documents(load_data()) instead of get_vector_store()
    classify()
    summarize_by_product()
    summarize_by_brand()

