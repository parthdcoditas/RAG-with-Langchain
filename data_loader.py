from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_wikipedia( chunk_size=500, chunk_overlap=50):
    docs = WikipediaLoader(query="Python (programming language)", load_max_docs=2).load()
    print(len(docs))
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
        
    chunks = text_splitter.split_documents(docs)
    return chunks



