from app.repo.document_json_db import DocumentJsonDB
from app.services.wikipedia_loader import WikipediaLoader
import os

def download_docs(url,markdown_dir,json_dir):
    loader = WikipediaLoader(url)
    loader.save_to_document_json(os.path.join(json_dir, f"{loader.uuid}.json"))
    loader.save_markdown(os.path.join(markdown_dir, f"{loader.uuid}.md"))
    # print(f"Document '{loader.title}' has been downloaded successfully.")


def load_doc_db(docs_db:DocumentJsonDB):
    print(docs_db.list()[:6])

def load_document(url,docs_db:DocumentJsonDB):
    return docs_db.find_by_url(url)


def main():
    url="https://en.wikipedia.org/wiki/List_of_tallest_buildings_in_New_York_City"
    json_dir="assets/dataset/document_json"
    markdown_dir="assets/dataset/markdown_files"

    docs_db=DocumentJsonDB(json_dir)

    download_docs(url,markdown_dir,json_dir)
    load_doc_db(docs_db)
    doc=load_document(url,docs_db)
    print(f"Document loaded: {doc.metadata['title']}")


if __name__ == "__main__":
    main()