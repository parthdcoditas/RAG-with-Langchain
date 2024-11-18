import psycopg2
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import DistanceStrategy
from langchain_community.vectorstores import PGVector

def store_embeddings_in_pgvector(documents, db_config):
    
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    connection_string = f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
    print('ssssssss')
    pgvector = PGVector.from_documents(
        documents=documents,
        embedding=embeddings_model,
        collection_name="embeddings_collection",
        distance_strategy=DistanceStrategy.COSINE,
        connection_string=connection_string,
        collection_metadata={"description": "Document embeddings"},
        pre_delete_collection=True,
        use_jsonb=True
    )
    
    return pgvector


