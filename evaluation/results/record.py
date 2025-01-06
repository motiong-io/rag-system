SAVE_DIR = 'evaluation/results/'

class ResultRecordService:
    def __init__(self,save_dir):
        self.save_dir = save_dir
    
    def add(self, result):
        self.record.append(result)
    
