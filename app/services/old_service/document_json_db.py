from app.model.document_model import DocumentJSON
import os
import json
from app.utils.md5hash import md5hash

class DocumentJsonDB:
    def __init__(self, dir_path: str):
        self.dir_path = dir_path

    def save(self, document: DocumentJSON):
        document.save_json(f"{self.dir_path}/{document.metadata['uuid']}.json")

    def list(self):
        return os.listdir(self.dir_path)

    def load(self, uuid: str) -> DocumentJSON:
        file_name = f"{self.dir_path}/{uuid}.json"
        if os.path.isfile(file_name):
            with open(file_name, "r",encoding='utf-8') as f:
                data = json.load(f)
                return DocumentJSON.from_dict(data)
        else:
            return None
    def find_by_url(self, url: str) -> DocumentJSON:
        uuid = md5hash(url)
        return self.load(uuid)

    def delete(self, file_name: str):
        try:
            if os.path.isfile(file_name):
                os.remove(file_name)
                print(f"File '{file_name}' has been deleted successfully.")
            else:
                print(f"File '{file_name}' does not exist.")
        except Exception as e:
            print(f"An error occurred while trying to delete the file: {e}")