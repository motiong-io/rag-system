# Constants for directory paths
MARKDOWN_DIR = "assets/dataset/markdown_files"
DOCUMENT_DIR = "assets/dataset/document_json"
EMBEDDINGS_DIR = "assets/dataset/embeddings_list"

# Function to get directory paths based on save flags
def get_directories(save_markdown, save_document, save_embeddings):
    markdown_dir = MARKDOWN_DIR if save_markdown else None
    document_dir = DOCUMENT_DIR if save_document else None
    embeddings_dir = EMBEDDINGS_DIR if save_embeddings else None
    return markdown_dir, document_dir, embeddings_dir

WEAVIATE_COLLECTION_NAME = "ContextualVectors"


