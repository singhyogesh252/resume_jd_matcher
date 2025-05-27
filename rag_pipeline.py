import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
# from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.memory import ConversationBufferMemory
# from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
import tempfile
from langchain_core.documents import Document



# load_dotenv()
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def load_uploaded_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PDFPlumberLoader(tmp_path)
    pages = loader.load()
    return pages

def create_vector_store(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

def match_resume_to_jd(resume_text, jd_vector_store, top_k=5):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    query_embedding = embeddings.embed_query(resume_text)
    docs_and_scores = jd_vector_store.similarity_search_by_vector(query_embedding, k=top_k)
    return docs_and_scores

def main(uploaded_file,jd_text):
    resumes = load_uploaded_pdf(uploaded_file)
    jd_doc = Document(page_content=jd_text)
    jd_vector_store = create_vector_store([jd_doc])

    # retriever = jd_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
#     memory = ConversationBufferMemory(
#         memory_key="chat_history",
#         return_messages=True
#     )

#     llm = ChatOpenAI(
#     model_name="gpt-3.5-turbo",  # or "gpt-4"
#     temperature=0.3,
#     max_tokens=512
# )


#     # Conversational QA prompt with memory
#     chat_prompt = ChatPromptTemplate.from_messages([
#         SystemMessagePromptTemplate.from_template(
#             "You are an AI assistant specialized in resume and job description analysis. "
#             "Use the following context to inform your responses:\n{context}"
#         ),
#         MessagesPlaceholder(variable_name="chat_history"),
#         HumanMessagePromptTemplate.from_template("{question}")
#     ])

#     qa_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=retriever,
#         memory=memory,
#         combine_docs_chain_kwargs={"prompt": chat_prompt},
#         return_source_documents=True
#     )

    return resumes, jd_doc, jd_vector_store