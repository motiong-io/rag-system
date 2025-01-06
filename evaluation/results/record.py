SAVE_DIR = 'evaluation/results/'

class ResultRecordService:
    def __init__(self,file_name:str, save_dir:str=SAVE_DIR):
        self.save_path = save_dir + file_name
        
    
    def add(self, result):
        self.record.append(result)
    
