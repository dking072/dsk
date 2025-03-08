import os
import glob
import argparse
import pickle

#Parsing- get root script
parser = argparse.ArgumentParser()
parser.add_argument("-g","--debug",action="store_true")
parser.add_argument("root_script")
args = parser.parse_args()

def load_pkl(filename):
    with open(filename,"rb") as file: #
        return pickle.load(file)

#Set params to change and values
#Important that they are typed correctly!!! 

#DMRG CONSTRAINED
params_to_change = {
    "NUM_H":[8,16,32],
    "NUM_H_PER_FRAG":[2],
    "DIST":[1.54],
}

#Key in special exemptions if you want:
def check_special_rules(name):
    return True

#Also need to tell it how to write the name:
#Needs to be in order of above:

#DMRG CONSTRAINED 
def get_name(NUM_H,NUM_H_PER_FRAG,DIST):
    fn = f"ccircle{NUM_H}_frag{NUM_H_PER_FRAG}_d{int(DIST*100)}.py"
    return fn

root_script = args.root_script
with open(root_script,"r") as file:
    original_lines = file.readlines()

def get_lines(iter_params):
    new_lines = []
    for line in original_lines:
        for k in iter_params.keys():
            if k in line:
                val = iter_params.pop(k) 
                if type(val) is str:
                    new_lines += k + " = " + "\"" + val + "\"" + "\n"
                else:
                    new_lines += k + " = " + str(val) + "\n"
                break
        else:
            new_lines += line
    return new_lines

from itertools import product

#Iterate over product of iterations!
for tple in product(*params_to_change.values()):
    print(tple)
    # print(tple)
    lines = get_lines(dict(zip(params_to_change.keys(),tple)))
    name = get_name(*tple)
    if name != root_script and not args.debug:
        if check_special_rules(name):
            with open(name,"w+") as file:
                for line in lines:
                    file.write(line)


