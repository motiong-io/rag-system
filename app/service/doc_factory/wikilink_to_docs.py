from app.service.doc_factory.data_model import Document
from langchain_community.document_loaders import WikipediaLoader
from urllib.parse import unquote
import json
import os

def get_wikipedia_title(url):
    # Split the URL to get the last part after "/wiki/"
    title_part = url.split('/wiki/')[-1]
    # Decode any percent-encoded characters, e.g., spaces represented as %20
    title = unquote(title_part)
    # Replace underscores with spaces if needed
    title = title.replace('_', ' ')
    return title


def link_to_json_file(wiki_link: str, language: str):
    wiki_query = get_wikipedia_title(wiki_link)
    try:
        pages = WikipediaLoader(query=wiki_query.strip(), lang=language, load_all_available_meta=False).load()
        file_name = wiki_query.strip()
        return file_name, pages
    except Exception as e:
        job_status = "Failed"
        message = "Failed To Process Wikipedia Query"
        error_message = str(e)
        file_name = wiki_query.strip()
        return file_name, {
            "job_status": job_status,
            "message": message,
            "error": error_message,
            "file_name": file_name
        }


def wikilink_to_json_docs(wiki_link:str,save_dir:str):
    print(wiki_link)
    file_name, pages = link_to_json_file(wiki_link, 'en')
    file_name = file_name.replace(" ", "_")
    combined_content = "\n".join([page.page_content for page in pages])
    title=pages[0].metadata['title']
    print(title)
    # Ensure the directory exists
    os.makedirs(f'./{save_dir}/', exist_ok=True)

    # Save the pages to a file
    with open(f'./{save_dir}/{file_name}.json', 'w') as f:
        content=Document(
            page_content=combined_content,
            metadata= {'title':pages[0].metadata['title'],'url':wiki_link}
        )
        json.dump(content, f)

    print(f"Saved pages to /{save_dir}/{file_name}.json")

