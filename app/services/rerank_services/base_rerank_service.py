from abc import ABC, abstractmethod

class BaseRerankService(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def rerank(self, corpus, query, top_n):
        """
        This method should be overridden by subclasses to implement specific reranking logic.
        
        :param corpus: The corpus of documents to be considered for reranking
        :param query: The query to rerank the documents against
        :param top_n: The number of top documents to return after reranking
        :return: List of reranked documents
        """
        pass