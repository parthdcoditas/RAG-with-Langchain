from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from data_loader import load_and_split_wikipedia
from embedding_store import store_embeddings_in_pgvector
# from retrieval_qa import initialize_qa_chain, get_response
from dotenv import load_dotenv
from langchain_postgres import PostgresChatMessageHistory
from langchain.schema.messages import BaseMessage
import uuid
import os
import psycopg2

load_dotenv()

db_config = {
    "dbname": os.getenv("dbname"),
    "user": os.getenv("username"),
    "password": os.getenv("password"),
    "host": os.getenv("host"),
    "port": os.getenv("port")
}

if __name__ == "__main__":
    print("Loading and splitting data...")
    chunks = load_and_split_wikipedia()

    print("Generating embeddings and storing in PGVector...")
    pgvector = store_embeddings_in_pgvector(chunks, db_config)

    retriever = pgvector.as_retriever()
    llm = ChatGroq(api_key=os.getenv("groq_api_key"))
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_template("""Answer the following question based on the provided context:
    
    Context: {context}
    Chat History: {chat_history}
    Question: {question}
    
    Answer:""")

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "chat_history": lambda x: x.get("chat_history", "")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    connection = psycopg2.connect(
        dbname=db_config["dbname"],
        user=db_config["user"],
        password=db_config["password"],
        host=db_config["host"],
        port=db_config["port"]
    )
    session_id = str(uuid.uuid4())
    print(session_id)
    chat_history = PostgresChatMessageHistory(
        "chat_history",
        session_id,
        sync_connection=connection
    )

    while True:
        user_query = input("Enter your query (or type 'exit' to quit): ").strip()
        if user_query.lower() == "exit":
            print("The end!")
            break

        # Simplify chat history handling
        try:
            history_messages = []
            for message in chat_history.messages:
                history_messages.append(f"{message.type}: {message.content}")
            history_str = "\n".join(history_messages[-5:])  # Get last 5 messages
        except Exception:
            history_str = ""  # Fallback if there's an error getting history

        print("Retrieving and generating response...")

        response = rag_chain.invoke(
            user_query
        )
        print(f"Response: {response}")

        chat_history.add_message(HumanMessage(content=user_query))
        chat_history.add_message(AIMessage(content=response))
        