from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

def initialize_qa_chain(pgvector, groq_api_key):
    
    llm = ChatGroq(api_key=groq_api_key)
    
    prompt = ChatPromptTemplate.from_template("""You are an intelligent assistant. Use the provided context and conversation history to answer the question:

    Previous Conversation:
    {chat_history}
                                              
    Retrieval Context:
    {context}

    Current Question: {input}
    Do not give any other information in the response apart from the answer.
    Answer:""")
    
    document_chain = create_stuff_documents_chain(llm, prompt)

    qa_chain = create_retrieval_chain(pgvector.as_retriever(), document_chain)
    return qa_chain

def get_response(qa_chain, user_query, chat_history=None):
    history_context = ""
    if chat_history:
        history_context = "\n".join([f"{type(msg).__name__}: {msg.content}" for msg in chat_history])
    
    input_data = {
        "input": user_query,
        "chat_history": history_context
    }

    response = qa_chain.invoke(input_data)
    return response["answer"]
