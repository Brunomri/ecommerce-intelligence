from typing import List, Optional
from langchain.schema import Document
from langchain_community.document_loaders.csv_loader import CSVLoader
import pandas as pd


class CustomCSVLoader(CSVLoader):
    """Create a custom CSV loader to limit the maximum number of CSV rows to load"""
    def __init__(
        self,
        file_path: str,
        csv_args: Optional[dict] = None,
        metadata_columns: Optional[List[str]] = None,
        max_rows: Optional[int] = None,
    ):
        super().__init__(file_path=file_path, csv_args=csv_args, metadata_columns=metadata_columns)
        self.max_rows = max_rows

    def load(self) -> List[Document]:
        # Load only the first `max_rows` rows if specified
        df = pd.read_csv(self.file_path, **(self.csv_args or {}), nrows=self.max_rows)

        documents = []
        for i, row in df.iterrows():
            metadata = {"source": self.file_path, "row": i}
            if self.metadata_columns:
                metadata.update({col: row[col] for col in self.metadata_columns if col in row})

            # Format the row as 'key: value\n' for each column
            content = "\n".join(f"{k}: {v}" for k, v in row.items())
            documents.append(Document(page_content=content, metadata=metadata))

        return documents
