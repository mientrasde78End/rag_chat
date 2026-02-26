from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag.vectorstore import get_vectorstore


def get_rag_chain():
    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        """
        Responde la pregunta usando SOLO el contexto dado.
        Si no sabes la respuesta, di "No lo sé según los documentos".

        Contexto:
        {context}

        Pregunta:
        {question}
        """
    )

    llm = ChatOllama(
        model="mistral",   # o "llama3.1"
        temperature=0
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": lambda x: x
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain