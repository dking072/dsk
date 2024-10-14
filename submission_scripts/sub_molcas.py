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

#Parser
parser = argparse.ArgumentParser(description="Submit a bash file to queue that sets PySCF memory and runs the python script")

#Resources
parser.add_argument("-q", "--queue", default="amdsmall", help="Queue to submit to  (default: %(default)s)")
parser.add_argument("-p", "--procs", default=1, type=int,help="Number of processors per node (default: %(default)s)") #Testing defaults :)
parser.add_argument("-t", "--time", default=95, type=int, help="Number of hours (default: %(default)s)")
parser.add_argument("-m", "--mem", default=2, type=int, help="Memory in GB (default: %(default)s)")

#Options
parser.add_argument("-g", "--dont_submit", action="store_true", help="Only write pbs files (don't submit to queue, useful for testing) (default: %(default)s)")
parser.add_argument("-c", "--confirm", action="store_true", help="Confirm settings before submission (default: %(default)s)")
parser.add_argument("-ks", "--keep_scripts", action="store_true", help="Copy scripts into directories instead of moving on submission")

#Scripts and parse arguments
parser.add_argument("script_names", nargs="+", help="Scripts to run tests on (required)")
args = parser.parse_args()

def main():
    if args.confirm:
        confirm_settings()

    #Write inputs and pbs files
    for script_name in args.script_names:
        method_name = script_name.split(".")[0]
        write_slurm(method_name)
        submit(method_name)

def confirm_settings():
    print("Submission overview:")
    print("Queue: ", str(args.queue))
    print("Processors: ", str(args.procs))
    print("Walltime: ", str(args.time) + " hours")
    print("RAM: ", str(args.mem) + " GB")
    # print("Data: ", args.dataset)
    # print("Method: ", method_name)
    print("Keep scripts: ", str(args.keep_scripts))
    #print("Parallel: ", str(args.parallel))

    answer=input("Are these settings okay? (y/n)\n")
    if answer=='y':
        print("Great!")
    else:
        print("Okay, exiting...")
        exit(1)

def write_slurm(method_name):

    slurm_name = method_name + ".slurm"

    init_lines = [
    "#!/bin/bash\n\n",
    "#SBATCH --nodes=1\n",
    "#SBATCH --account=pi-lgagliardi\n",
    f"#SBATCH --ntasks-per-node={args.procs}\n",
    f"#SBATCH --time={args.time}:00:00\n",
    f"#SBATCH --mem={args.mem}G\n",
    f"#SBATCH --partition={args.queue}\n",
    f"#SBATCH --error={method_name}.e\n",
    f"#SBATCH --output={method_name}.log\n",
    "\n"
    ]

    mem_mb = int(args.mem*1000)
    safety = 500*int(args.procs)
    memper = int((mem_mb - safety)/int(args.procs))

    with open(slurm_name,'w+') as f:
        for line in init_lines:
            f.write(line)

        f.write("#MODULES\n")
        # f.write("module load python" + "\n")
        f.write("\n")

        f.write("#PATHING\n")
        f.write("cd $SLURM_SUBMIT_DIR\n")
        f.write("mkdir -p $SCRATCH/$USER/$SLURM_JOBID\n")
        f.write("\n")

        f.write("#ENVIRONMENT\n")
        #f.write("export MOLCAS=/project/lgagliardi/shared/Apps/OpenMolcas/builds/parallel\n")
        f.write("explort MOLCAS=/project/lgagliardi/shared/Apps/OpenMolcas/builds/serial\n")
        f.write("export MOLCAS_WORKDIR=$SCRATCH/$USER/$SLURM_JOBID/\n")
        f.write("export MOLCAS_CURRDIR=$SLURM_SUBMIT_DIR\n")
        #f.write("export MOLCAS_NPROCS=$SLURM_NTASKS_PER_NODE\n")
        f.write(f"export MOLCAS_MEM={memper}\n")
        f.write("export OMP_NUM_THREADS=1\n")
        f.write("\n")

        f.write("#EXECUTION\n")
        f.write(f"/project/lgagliardi/shared/bin/pymolcas -f {method_name}.inp\n")
        f.write("\n")

def submit(method_name):
    #Directory setup
    dir_name = method_name
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
    script_name = f"{method_name}.inp"
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

#Things that need to be done for molcas approaches:
# f.write("#FILE STRUCTURE\n")
# f.write("mkdir -p " + "/" + scratch + "/$USER/$PBS_JOBID\n")
# f.write("cd $PBS_O_WORKDIR\n\n")
# f.write("export MOLCAS_WORKDIR=" + "/" + scratch + "/$USER/$PBS_JOBID" + "\n") #set scratch
# f.write("export MOLCAS_CURRDIR=$PBS_O_WORKDIR" + "\n") #set scratch
# f.write("export MOLCAS_PROJECT=" + name + "\n") #Already defaults to this

#REFERENCE: MOLCAS ENVIRONMENT VARIABLES:
#Important Environment Variables (can be displayed with pymolcas -env)
#MOLCAS_PROJECT (defaults to input name)
#MOLCAS_WORKDIR (scratch directory)
#MOLCAS_CURRDIR (location of input, default for all outputs)
#MOLCAS (location of molcas build)
#MOLCAS_NPROCS (used for parallel, defines # of procs to use)
#MOLCAS_MEM (memory for work array, in mb; defaults to 1024) (maybe possible to use gb/tb? not sure) #This is RAM, not DISK!

#Extras:
#MOLCAS_COLOR (uses markup characters? huh.)
#MOLCAS_DISK (in mb, defaults to 2gb? should increase this)
#MOLCAS_ECHO_INPUT (control echoing of input?)
#MOLCAS_INPORB_VERSION (version for orbital files?)
#MOLCAS_NEW_WORKDIR=YES (cleans scratch before use)
#MOLCAS_KEEP_WORKDIR (to keep scratch)
#MOLCAS_LICENSE (specifies directory with license? lol)
#MOLCAS_MAXMEM (hard limit for memory, defaults to molcas_mem)
#MOLCAS_MOLDEN (generate molden file if set to on (default?))
#MOLCAS_OUTPUT (save to a specified directory, defaults to input directory)
#MOLCAS_PRINT (SILENT, TERSE, NORMAL, VERBOSE, DEBUG, INSANE) lol
#MOLCAS_PROJECT (see above, defaults to input name)
#MOLCAS_REDUCE_PRT (set to NO, print level in DO WHILE loop is not reduced)
#MOLCAS_REDUCE_NG_PRT (print level in numerical gradient not reduced)
#MOLCAS_SAVE (alter default filenames?)
#MOLCAS_TIME (if set, swicth on timing information?)
#MOLCAS_TIMELIM (set up time limit for each module)
#MOLCAS_TRAP (if off, continue even if non-zero/one return code returned? huh.)
#MOlCAS_VALIDATE (YES- check each module, CHECK-will validate first, but contine?, FIRST- valicate before running anything?)
#MOLCAS_WORKDIR (scratch, see above)



