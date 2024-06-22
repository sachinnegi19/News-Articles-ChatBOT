import os
import pickle
import dill
import pinecone
from langchain_openai import ChatOpenAI


os.environ['OPENAI_API_KEY'] = 'sk-proj-ewXqd1r8ITfcge9E65x4T3BlbkFJf4TaqeodcRCgWOvxn7MV'
llm = ChatOpenAI(api_key="sk-proj-ewXqd1r8ITfcge9E65x4T3BlbkFJf4TaqeodcRCgWOvxn7MV" )


from langchain_community.document_loaders import UnstructuredURLLoader
loader: UnstructuredURLLoader = UnstructuredURLLoader(["https://economictimes.indiatimes.com/markets/stocks/news/nifty-looks-poised-for-a-bullish-run-next-month-heres-why/articleshow/110413404.cms?from=mdr" ,
                                "https://economictimes.indiatimes.com/industry/banking/finance/banking/rbi-approves-appointment-of-pradeep-kumar-sinha-as-part-time-chairman-of-icici-bank/articleshow/110412028.cms"])
# data = loader.load()

# Scrapping the data from URLs
from langchain_community.document_loaders import UnstructuredURLLoader
loader = UnstructuredURLLoader(["URL1" , "URL2"])
data = loader.load()

# Splitting the Data into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n', '.', ','],
    chunk_size=1000,
)
docs = splitter.split_documents(data)

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

from langchain_community.vectorstores import Chroma
persist_directory="chroma_db"
vectordb= Chroma.from_documents(documents=docs , embedding=embeddings
                                , persist_directory=persist_directory)




our_query = "USER QUERY"

db=Chroma(persist_directory=persist_directory , embedding_function= embeddings)
matching_docs = db.similarity_search_with_score(our_query , k=2)

from langchain.chains import RetrievalQA
retriever = db.as_retriever()
chain = RetrievalQA.from_chain_type(llm , chain_type="stuff" , retriever=retriever)

answer = chain.invoke(our_query)
print(answer)

# from pinecone import Pinecone
# from langchain_pinecone import PineconeVectorStore
#
# pinecone = Pinecone(
#     api_key="cd690b36-b166-48f6-8501-eb32809136f2",
#     envirnment="us-east-1"
# )
# index_name = "langchainproj"
#
# index = PineconeVectorStore.from_documents(documents=docs, index_name=index_name , embedding=embeddings)
#
# def retrieve_query(query , k=3):
#     matching_results = index.query(vector=embeddings.embed_query(query), top_k=k)
#     return matching_results
#
# from langchain.chains.question_answering import load_qa_chain
#
# chain = load_qa_chain(llm , chain_type="stuff")
#
# def retrieve_answers(query):
#     response = chain.run(input_documents=retrieve_query(query) , question=query)
#     return response
#
# our_query = "who is the chairman of icici-bank?"
#
# answers = retrieve_answers(our_query)
# print(answers)

# from langchain_community.vectorstores import FAISS
#
# vectorindex_openai = FAISS.from_documents(docs, embeddings)
#
# vectorindex_openai = vectorindex_openai.serialize_to_bytes()
#
# file_path="vector_index.pkl"
# with open(file_path, "wb") as f:
#      pickle.dump(vectorindex_openai,f)
#
# if os.path.exists(file_path):
#     with open(file_path, "rb") as f:
#         vectorindex = pickle.load(f)
#         vectorindex=FAISS.deserialize_from_bytes(vectorindex , embeddings=embeddings)
#
#
# # from langchain_community.retrievers import MultiQueryRetriever
# from langchain.retrievers.multi_query import MultiQueryRetriever
# chain = MultiQueryRetriever.from_llm(llm=llm, retriever = vectorindex.as_retriever())
#
# query = "who is the chairman of icici bank?"
#
# # langchain.debug=True
#
# print(chain.invoke(query))









