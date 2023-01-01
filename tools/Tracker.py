import numpy as np

from tools.FileManager import make_directory

# TODO: add automatic check for former session by checking if a directory exist - and if so, load the progress...
class Tracker:
    
    def __init__(self, prefix="SUMMERY", dir_path=None) -> None:
        
        if dir_path is None:
            self.dir_path   =   make_directory(prefix=prefix)
        else:
            self.dir_path   =   dir_path
            
        self.file_name      =   "output_summary.npy"
        
    def save(self, summary : dict):
        dict_path = self.dir_path + "\\" + self.file_name
        np.save(dict_path, summary)
        
    def load(self):
        dict_path = self.dir_path + "\\" + self.file_name
        summary = np.load(dict_path, allow_pickle=True)
        return summary.item()
    
    def file_path(self):
        return self.dir_path + "\\" + self.file_name
    
    def directory_path(self):
        return self.dir_path