import pickle

def dump_pkl(obj,fn):
    with open(fn,"wb+") as file:
        pickle.dump(obj,file)
        
def load_pkl(fn):
    with open(fn,"rb") as file:
        return pickle.load(file)