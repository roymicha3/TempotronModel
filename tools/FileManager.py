import os
from datetime import datetime

def get_full_path(relative_path : str) -> str:
    absolute_path = os.path.dirname(__file__)
    absolute_path = os.path.join(absolute_path, "..\\results")
    full_path = os.path.join(absolute_path, relative_path)
    
    return full_path

def make_directory(directory = None, prefix = "RESULTS", session_code = None):
    
    if directory is None:
        if session_code is None:
            directory = generate_dir_name(prefix=prefix)
            
        else:
            directory = get_dir_name(prefix, session_code)

    path = get_full_path(directory)

    os.mkdir(path)
    print("Directory '% s' created" % directory)
    
    return path
    
def generate_dir_name(prefix):
    
    current_datetime = datetime.now()
    print("Current date & time : ", current_datetime)
    
    # convert datetime obj to string
    dir_name = prefix + str(current_datetime).replace(' ', '_').replace(':', '-')

    return dir_name

def get_dir_name(prefix, session_code):
    
    dir_name = prefix + session_code

    return dir_name

def is_dir_exist(prefix, session_code):
    dir_name = get_dir_name(prefix, session_code)
    path = get_full_path(dir_name)
    
    return os.path.exists(path)