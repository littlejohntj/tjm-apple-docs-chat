from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS

DATA_PATH = "data/developer_apple_com"
DB_FAISS_PATH = "vectorstores/db_faiss"


# create vector database
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH, glob="*/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        None, True, chunk_size=500, chunk_overlap=50
    )
    texts = text_splitter.split_documents(documents)

    # t = [text.page_content for text in texts]

    # embeddings = OpenAIEmbeddings(
    #     openai_api_key="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",
    #     model="text-embedding-ada-002",
    #     show_progress_bar=True,
    #     deployment="tj-and-mike",
    # )

    # e = embeddings.embed_documents(t, chunk_size=500)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)


if __name__ == "__main__":
    create_vector_db()
