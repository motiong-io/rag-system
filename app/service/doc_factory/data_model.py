from dataclasses import dataclass, field
@dataclass
class Document:
    page_content: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        # Initialize metadata title and url if not present
        self.metadata.setdefault('title', '')
        self.metadata.setdefault('url', '')

    def to_dict(self):
        return {
            'page_content': self.page_content,
            'metadata': self.metadata
        }