from langchain_text_splitters import RecursiveCharacterTextSplitter

from data_processing.CustomCSVLoader import CustomCSVLoader


def load_data():
    """Read CSV file and split it into chunks."""
    loader = CustomCSVLoader(file_path="data/B2W-Reviews01.csv", max_rows=300)
    documents = loader.load()
    chunks = split_documents(documents)
    return chunks

def split_documents(documents):
    """Splits CSV data into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    all_splits = text_splitter.split_documents(documents)
    print(f"Split CSV data into {len(all_splits)} chunks")
    return all_splits