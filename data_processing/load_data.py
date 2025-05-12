from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter

# Read CSV file and split it into chunks
def load_data():
    loader = CSVLoader(file_path="data/B2W-Reviews01.csv", csv_args={'delimiter': ','})
    documents = loader.load()
    chunks = split_documents(documents)
    return chunks

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    all_splits = text_splitter.split_documents(documents)
    print(f"Split CSV data into {len(all_splits)} chunks")
    return all_splits