import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()


st.title("News research tool 📈")
st.sidebar.title("News article URLs")

urls= []
for i in range(3):
    url = st.sidebar.text_input(f"Enter article URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI()
persist_directory = "chroma_db"

if(process_url_clicked):
    # load the data of urls
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    #split the data
    splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = splitter.split_documents(data)

    #create embeddings
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)

query = main_placeholder.text_input("Question : ")
if query:
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    matching_docs = db.similarity_search_with_score(query, k=2)
    retriever = db.as_retriever()
    chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)
    answer = chain.invoke(query)

    st.header("Answer")
    st.write(answer["result"])

    sources = answer.get("sources", "")
    if sources:
        st.subheader("Sources:")
        sources_list = sources.split("\n")  # Split the sources by newline
        for source in sources_list:
            st.write(source)
