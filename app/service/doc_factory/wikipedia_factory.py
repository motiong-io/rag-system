import logging
from langchain_community.document_loaders import WikipediaLoader


class WikipediaFactory:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_documents(self, wiki_query: str, language: str):
        try:
            pages = WikipediaLoader(query=wiki_query.strip(), lang=language, load_all_available_meta=False).load()
            file_name = wiki_query.strip()
            self.logger.info(f"Total Pages from Wikipedia = {len(pages)}")
            return file_name, pages
        except Exception as e:
            job_status = "Failed"
            message = "Failed To Process Wikipedia Query"
            error_message = str(e)
            file_name = wiki_query.strip()
            self.logger.error(f"Failed To Process Wikipedia Query: {file_name}")
            self.logger.exception(f'Exception Stack trace: {error_message}')
            return file_name, {
                "job_status": job_status,
                "message": message,
                "error": error_message,
                "file_name": file_name
            }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    factory = WikipediaFactory()
    query = "Python programming language"
    language = "en"
    file_name, result = factory.get_documents(query, language)
    if isinstance(result, dict) and result.get("job_status") == "Failed":
        print(f"Error: {result['message']}")
        print(f"Details: {result['error']}")
    else:
        print(f"Successfully retrieved {len(result)} pages for query '{query}'")
        print(result[0].page_content)