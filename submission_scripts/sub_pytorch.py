#!/usr/bin/env python3

#########################################
####### PYSCF SUBMISSION SCRIPT########
############ Daniel King ################
############# 2/24/2021 #################
#########################################

#Imports
import argparse
import os

#Global vars
del_all = False

cpu_to_gpu_dct = {
    "1080TI":2,
    "GTX2080TI":2,
    "TITAN":4,
    "V100":4,
    "A40":8,
    "A5000":4,
}

#Parser
parser = argparse.ArgumentParser(description="Submit a bash file to queue that sets PySCF memory and runs the python script")

#Resources
parser.add_argument("-q", "--queue", default="savio4_gpu", help="Partition to submit to  (default: %(default)s)")
parser.add_argument("-g", "--gpus", default=1, type=int,help="Number of gpus to request (default: %(default)s)")
parser.add_argument("-gtyp", "--gpu_type", default="A5000", help="GPU type to request (default: %(default)s)")
parser.add_argument("-t", "--time", default=72, type=int, help="Number of hours (default: %(default)s)")
parser.add_argument("-qos", "--qos", default="savio_lowprio", help="Number of hours (default: %(default)s)")
#other options -- 12monkeys_gpu4_normal

#Options
parser.add_argument("-test", "--dont_submit", action="store_true", help="Only write pbs files (don't submit to queue, useful for testing) (default: %(default)s)")
parser.add_argument("-c", "--confirm", action="store_false", help="Confirm settings before submission (default: %(default)s)")
parser.add_argument("-ks", "--keep_scripts", action="store_true", help="Copy scripts into directories instead of moving on submission")
parser.add_argument("-r", "--requeue", action="store_true", help="Whether to requeue the job -- only use if your job is restartable!")

#Scripts and parse arguments
parser.add_argument("script_names", nargs="+", help="Scripts to run tests on (required)")
args = parser.parse_args()

def main():
    if args.confirm:
        confirm_settings()

    #Write inputs and pbs files
    for script_name in args.script_names:
        print(f"Submitting {script_name}...")
        method_name = script_name.split(".")[0]
        print(method_name)
        write_slurm(method_name)
        submit(method_name)

def confirm_settings():
    print("Submission overview:")
    print("Queue: ", str(args.queue))
    print("QOS: ",str(args.qos))
    print("GPUs: ", str(args.gpus))
    print("Walltime: ", str(args.time) + " hours")
    print("Keep scripts: ", str(args.keep_scripts))
    print("Requeue: ", str(args.requeue))

    answer=input("Are these settings okay? (y/n)\n")
    if answer=='y':
        print("Great!")
    else:
        print("Okay, exiting...")
        exit(1)

def write_slurm(method_name):

    slurm_name = method_name + ".slurm"
    procs = cpu_to_gpu_dct[args.gpu_type]*args.gpus

    init_lines = [
    "#!/bin/bash\n\n",
    "#SBATCH --ntasks=1\n",
    "#SBATCH --account=co_12monkeys\n",
    f"#SBATCH --qos={args.qos}\n",
    f"#SBATCH --gres=gpu:{args.gpu_type}:{args.gpus}\n",
    f"#SBATCH --cpus-per-task={procs}\n",
    f"#SBATCH --time={args.time}:00:00\n",
    f"#SBATCH --partition={args.queue}\n",
    f"#SBATCH --error={method_name}.e\n",
    f"#SBATCH --output={method_name}.log\n",
    ]
    if args.requeue:
        linit_lines += [f"#SBATCH --requeue\n"]
    init_lines += ["\n"]

    with open(slurm_name,'w+') as f:
        for line in init_lines:
            f.write(line)

        f.write("#MODULES\n")
        f.write("module load python" + "\n")
        f.write("\n")

        f.write("#PATHING\n")
        f.write("cd $SLURM_SUBMIT_DIR\n") 
        f.write("\n")

        f.write("#ENVIRONMENT\n")
        # f.write(f"export PYSCF_MAX_MEMORY={mem_mb}\n")
        # f.write("mkdir /scratch/midway3/king1305/$SLURM_JOB_ID\n")
        # f.write("export PYSCF_TMPDIR=/scratch/midway3/king1305/$SLURM_JOB_ID\n")
        # f.write("export TMPDIR=/scratch/midway3/king1305/$SLURM_JOB_ID\n")
        #if ("amd" in args.queue) or ("hm" in args.queue):
            #Redirects pyscf to amd installation:
        #    f.write(f"PYTHONPATH=/home/king1305/Apps_amd/pyscf:$PYTHONPATH\n")
        f.write("\n")

        f.write("#EXECUTION\n")
        f.write(f"python3 {method_name}.py &> logfile.log\n")

def submit(method_name):
    #Directory setup
    dir_name = method_name
    print("dir name:",dir_name)
    try:
        os.mkdir(dir_name)
    except FileExistsError:
        global del_all
        if del_all:
            os.system("rm -r " + dir_name)
            os.mkdir(dir_name)
        else:
            answer = input("File exists: " + dir_name + ". Delete all existing directories in this run and force continue (y/n)?")
            if answer == 'y':
                print("Okay, sounds good.")
                del_all = True
                os.system("rm -r " + dir_name)
                os.mkdir(dir_name)
            else:
                print("Okay, exiting...")
                exit(1)

    #Move .slurm file into directory
    os.system(f"mv {method_name}.slurm {dir_name}")
    
    #Move or copy in script:
    script_name = f"{method_name}.py"
    if args.keep_scripts:
        os.system("cp {0} {1}".format(script_name,dir_name))
    else:
        os.system("mv {0} {1}".format(script_name,dir_name))

    #Move to directory
    os.chdir(dir_name)

    #Submit or don't
    if args.dont_submit:
        print("Didn't submit. The pbs files have been written and placed into the directories.")
    else:
        os.system(f"sbatch {method_name}.slurm")

    #Move back out
    os.chdir("..")

if __name__ == '__main__':
    main()

