import os
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from embedding_utils import get_embeddings


def load_pdf_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
    return documents

def create_store_index(documents,index_path):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = get_embeddings()
    store = FAISS.from_documents(
        documents=texts, 
        embedding=embeddings
        )
    os.makedirs(index_path, exist_ok=True)
    store.save_local(index_path)


def index_exists(index_path):
    return os.path.exists(os.path.join(index_path, "index.faiss"))


def build_index_if_missing(source_dir, index_dir, label, force_rebuild=False):
    if index_exists(index_dir) and not force_rebuild:
        print(f"{label} index already exists at {index_dir}. Skipping embedding.")
        return

    documents = load_pdf_from_folder(source_dir)
    if not documents:
        print(f"No PDF files found in {source_dir}. Nothing to index.")
        return

    create_store_index(documents, index_dir)
    print(f"{label} index created and saved to {index_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS indexes only when they do not already exist.")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuilding both indexes.")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    syllabus_dir = os.path.join(base_dir, "data", "syllabus")
    past_papers_dir = os.path.join(base_dir, "data", "past_questions")
    syll_index_dir = os.path.join(base_dir, "faiss_index", "syllabus_index")
    past_index_dir = os.path.join(base_dir, "faiss_index", "past_papers_index")

    build_index_if_missing(syllabus_dir, syll_index_dir, "Syllabus", force_rebuild=args.rebuild)
    build_index_if_missing(past_papers_dir, past_index_dir, "Past papers", force_rebuild=args.rebuild)

