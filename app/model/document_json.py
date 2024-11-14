import json
from app.utils.md5hash import md5hash


class DocumentJSON:
    def __init__(self, page_content: str, title: str, url: str):
        self.page_content = page_content
        self.metadata = {
            "uuid": md5hash(url),
            "title": title,
            "url": url
        }

    def to_dict(self):
        return {
            "page_content": self.page_content,
            "metadata": self.metadata,
        }
    
    def save_json(self, filename: str):
        with open(filename, "w",encoding='utf-8') as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_dict(cls, data):
        return cls(
            page_content=data.get("page_content"),
            title=data.get("metadata")['title'],
            url=data.get("metadata")['url']
        )

