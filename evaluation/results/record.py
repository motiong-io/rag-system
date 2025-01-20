import os
from datetime import datetime

from evaluation.request_model import EvaluateRagRequest
SAVE_DIR = 'evaluation/results/20250120/rag-optimization-50-20'

class ResultRecordService:
    def __init__(self,eva_request:EvaluateRagRequest,base_path=SAVE_DIR):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.folder_path = os.path.join(base_path, self.timestamp)

        os.makedirs(self.folder_path, exist_ok=True)
        print(f"Folder created: {self.folder_path}")
        with open(os.path.join(self.folder_path, 'request.json'), 'w') as file:
            file.write(str(eva_request.model_dump_json()))

    def add_index_result(self, index:int, content:str):
        index_result_path = os.path.join(self.folder_path, f'{index}.txt')
        with open(index_result_path, 'a') as file:
            file.write(content)