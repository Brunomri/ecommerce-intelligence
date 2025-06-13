from rag.embeddings import get_vector_store
from rag.rag_chain import query_rag

def classify(result_path="results/classification/"):
    """Classify product reviews into topics and write output to file"""
    question = """
    Please extract the topics mentioned in the fields review_title and review_text for each review provided. 
    A few topic examples are: product quality, delivery, customer service, packaging and usage. 
    For each topic, add how many reviews belong to it. A review can belong to multiple topics. 
    Present the result in a table where each row has the topic and its review count, write the result in portuguese.
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

def summarize_by_category(result_path="results/summarization/category/"):
    """Summarize reviews by category and write output to file"""
    question = """
    Please summarize the reviews of all products using a combination of fields site_category_lv1 and site_category_lv2.
    Present the result in a table where each row has the fields site_category_lv1, site_category_lv2 and summary, write the result in portuguese.
    """
    query_rag(question, get_vector_store(), "summary_by_category", result_path)

def sentiment_analysis(result_path="results/sentiment/"):
    """Analyze sentiments by review and write output to file"""
    question = """
    Please consider the fields review_title, overall_rating, recommend_to_a_friend and review_text to build a 
    sentiment analysis for each review. Try to find balance when a low overall_rating is paired with a positive review_text or
    vice versa. Present the result in a table where each row has the fields product_id, product_name, review_title, overall_rating, a descriptive
    concise text about the sentiment analysis, and assign a sentiment category between negative, neutral or positive.
    Below the table also return the percentage of reviews for each sentiment category mentioned above. Write the results in portuguese.
    """
    query_rag(question, get_vector_store(), "sentiment_analysis", result_path)

def frequent_questions(result_path="results/questions/"):
    """Get the most frequent questions from the review texts"""
    question = """
    Please create a list of questions and answers based on the field review_text. Present the result in a table 
    where each row has the question and a possible response, write the result in portuguese. 
    """
    query_rag(question, get_vector_store(), "frequent_questions", result_path)

if __name__ == "__main__":
    # TODO: In case the index doesn't exist, create it by calling
    #  index_documents(load_data()) instead of get_vector_store()
    classify()
    summarize_by_product()
    summarize_by_brand()
    summarize_by_category()
    sentiment_analysis()
    frequent_questions()

