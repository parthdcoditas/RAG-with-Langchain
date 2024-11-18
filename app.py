import uuid
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_postgres import PostgresChatMessageHistory
from data_loader import load_and_split_wikipedia
from embedding_store import store_embeddings_in_pgvector
from retrieval_qa import initialize_qa_chain, get_response
from dotenv import load_dotenv
import psycopg
import os

load_dotenv()

db_config = {
    "dbname": os.getenv("dbname"),
    "user": os.getenv("username"),
    "password": os.getenv("password"),
    "host": os.getenv("host"),
    "port": os.getenv("port")
}

table_name = "chat_history"
session_id = str(uuid.uuid4()) 

if __name__ == "__main__":
    
    conn_info = f"dbname={db_config['dbname']} user={db_config['user']} password={db_config['password']} host={db_config['host']} port={db_config['port']}"
    sync_connection = psycopg.connect(conn_info)

    # Create the table schema for chat history if not exists
    PostgresChatMessageHistory.create_tables(sync_connection, table_name)
    chat_history = PostgresChatMessageHistory(table_name, session_id, sync_connection=sync_connection)

    print("Loading and splitting data...")
    chunks = load_and_split_wikipedia()

    print("Generating embeddings and storing in PGVector...")
    pgvector = store_embeddings_in_pgvector(chunks, db_config)

    print("Setting up QA chain...")
    groq_api_key = os.getenv("groq_api_key")
    qa_chain = initialize_qa_chain(pgvector, groq_api_key)

    while True:
        user_query = input("\nEnter your query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            print("The end!")
            break

        previous_messages = chat_history.messages[-5:] if len(chat_history.messages) >= 5 else chat_history.messages

        chat_history.add_messages([HumanMessage(content=user_query)])

        if previous_messages:
            history_content = "\n".join(
                f"{type(msg).__name__}: {msg.content}" for msg in previous_messages
            )

        print("Retrieving and generating response...")
        response = get_response(qa_chain, user_query, previous_messages)

        chat_history.add_messages([AIMessage(content=response)])

        print(f"Response: {response}")
