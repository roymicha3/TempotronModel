import numpy as np
import os

MAX_TAGS = 100

# TODO: add automatic check for former session by checking if a directory exist - and if so, load the progress...
class Tracker:
    
    def __init__(self, prefix="SESSION", tag=None) -> None:
        
        self.dir_path       =   ".\\results\\models\\"
        
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
            print("Directory '% s' created" % self.dir_path)
            
        if tag is None:
            tag = self.get_tag(self.dir_path, prefix)
            print(f"created a new tag: {tag} \n")
        
        self.file_name      =   prefix + f"-{tag}.npy"
        self.file_path      =   os.path.join(self.dir_path + self.file_name)
        
    def save(self, summary : dict):
        np.save(self.file_path, summary)
        
    def load(self):
        summary = np.load(self.file_path, allow_pickle=True)
        return summary.item()
    
    def get_file_path(self):
        return self.file_path
    
    def get_directory_path(self):
        return self.dir_path
    
    def session_exist(self):
        return os.path.exists(self.file_path)
    
    def get_tag(self, dir_path, prefix):
        # if the naming is changed, we need to remember to chage it here as well!
        for tag in range(MAX_TAGS):
            file_name      =   prefix + f"-{tag}.npy"
            file_path      =   os.path.join(dir_path + file_name)
            
            if (not os.path.exists(file_path)):
                return tag
            
        print("didnt fing a vacant tag! \n")
        assert(False)