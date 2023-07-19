import os
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import(
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
) 
os.environ['OPENAI_API_KEY']='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'
llm=OpenAI(temperature=1.0, verbose=True)
embeddings=OpenAIEmbeddings()
loader=PyPDFLoader('Deleuze-Nietzsche.pdf')
pages=loader.load_and_split()
store=Chroma.from_documents(pages, embeddings, collection_name='Deleuze-Nietzsche')
vector_store_info=VectorStoreInfo(
    name="Deleuze-Nietzsche",
    description="a philosophical paper as a pdf",
    vectorstore=store
)
toolkit=VectorStoreToolkit(vectorstore_info=vector_store_info)
agent_executor=create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)
st.title('Philosophical Paper Interpreter')
prompt=st.text_input('Put text here')

if prompt:
    response=agent_executor.run(prompt)
    st.write(response)
    with st.expander('Document Similarity Search'):
        search=store.similarity_search_with_score(prompt)
        st.write(search[0][0].page_content)
