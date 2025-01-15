SAVE_DIR = 'evaluation/results/'

class ResultRecordService:
    def __init__(self,file_name:str='results.txt', save_dir:str=SAVE_DIR):
        self.save_path = save_dir + file_name
        
    
    def add_contents(self, contents: str):
        with open(self.save_path, 'a') as file:
            file.write(str(contents) + '\n')
    
