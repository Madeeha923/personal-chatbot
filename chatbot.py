import os
import streamlit as st

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")



from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_groq_llm():
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",  # or "mixtral-8x7b-32768"
        temperature=0.9,
        max_tokens=1024,
        timeout=None,
        max_retries=2,
    )
    return llm

def main():
    st.title("Your personal Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Ask your Document")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                # Use the pieces of information provided in the context to answer user's question.
                # If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                # Dont provide anything out of the given context
                You are a document analysis assistant. Your task is to answer questions using ONLY the information provided in the context documents below. 

                STRICT RULES:
                - Use ONLY information from the provided context
                - Do NOT add any external knowledge or information
                - If the context doesn't contain enough information to answer fully, say "The provided documents do not contain sufficient information to answer this question completely"
                - Quote directly from the context when possible
                - Organize the information clearly and comprehensively using what's available in the context


                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """
        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain=RetrievalQA.from_chain_type(
                llm=load_groq_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response=qa_chain.invoke({'query':prompt})

            result=response["result"]
            source_documents=response["source_documents"]
            result_to_show=result+"\n\nSource Docs:\n"+str(source_documents)
            #response="Hi, I am MediBot!"
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")
if __name__ == "__main__":
    main()        