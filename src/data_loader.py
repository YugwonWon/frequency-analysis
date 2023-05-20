import os
import pickle

class DataLoader():
    """
    데이터 로딩 및 피클 저장

    """
    def __init__(self):
        self.src_data = {}
        
        
    def get_file_list(self, target):
        """ 
        파일을 불러온다.
        
        """     
        filenames = []
        try:
            for (path, _, files) in os.walk(target):
                for filename in files:
                    ext = os.path.splitext(filename)[-1]
                    if not filename.startswith('._'):
                        filenames.append(os.path.join(path, filename))
                        
        except PermissionError:
            pass
        
        return filenames



if __name__ == '__main__':
    data = DataLoader()


