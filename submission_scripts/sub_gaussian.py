#!/usr/bin/env python3

#Various Imports we need:
import argparse
import os
import sys
import shutil
import glob

#Argument Parsing
parser = argparse.ArgumentParser(description="sub_pyscf.py")

#Resource settings
parser.add_argument("-q", "--queue", default="caslake", help="Queue (default: %(default)s)")
parser.add_argument("-p", "--procs", default=1, type=int, help="Number of processors (per node) (default: %(default)s)")
parser.add_argument("-t", "--time", default=36, type=int, help="Walltime in hours (default: %(default)s)")
parser.add_argument("-m", "--mem", default=2, type=int, help="RAM in GB (default: %(default)s)")

#Technical settings, files
parser.add_argument("-v", "--version", default="g16", help="version of gaussian to use (default: %(default)s). Options: g09, g16.")
parser.add_argument("-g", "--debug", action="store_true", help="run in debug mode (default: %(default)s)")
parser.add_argument("-c", "--confirm", action="store_false", help="confirm settings (default: %(default)s)")
parser.add_argument("filenames", nargs="+", help="Gaussian input files to submit")

args = parser.parse_args()

#Confirm settings
print("Settings:")
print("Queue: ", args.queue)
print("Processors: ", args.procs)
print("Walltime (hours): ", args.time)
print("RAM: ", args.mem)
print("Version: ", args.version)

if args.confirm:
    ans=input("Are these settings alright? (y/n)\n")
    if ans == 'y':
        print("Great!")
    else:
        print("Okay, exiting...")
        exit(1)

now = vars(args)

#Write PBS
for filename in args.filenames:
    name = filename.split('.')[0]

    #Edit dictionary instead
    queue = now["queue"]
    procs =  now["procs"]
    time =  now["time"]
    mem =  now["mem"]
    version =  now["version"]
    mem = int(mem)

    #Prepare lines to write in pbs file
    user = os.environ['USER']

    lines = []

    lines += [
    "#!/bin/bash\n\n",
    "#SBATCH --account=pi-lgagliardi\n"
    "#SBATCH --ntasks=1\n",
    f"#SBATCH --cpus-per-task={procs}\n",
    f"#SBATCH --time={time}:00:00\n", #walltime
    f"#SBATCH --mem={mem}G\n", #RAM
    f"#SBATCH --partition={queue}\n",
    ]
    if "lgagliardi" in queue:
        lines += ["#SBATCH --qos=lgagliardi\n"]
    lines += [
    "#SBATCH --error={0}.e\n".format(name), #Error file
    "#SBATCH --output={0}.o\n".format(name), #Output file (from stdout)
    "\n"
    ]

    #Module loading
    lines += ["#MODULES\n"]
    #Gaussian version:
    if version == 'g16':
        lines += [
            "module load gaussian/16RevA.03\n"
        ]
    elif version == 'g09':
        lines += [
            "module load gaussian\n" #What Thais uses
        ]
    else:
        print("Requested version of Gaussian ({0}) not found!".format(version))
        exit(1)
    lines += ["\n"]

    #Set Gaussian environment variables
    lines += ["#ENVIRONMENT\n"]
    lines += [
    "export GAUSS_PDEF={0}\n".format(procs), #processors
    f"export GAUSS_MDEF={int(mem)-2}gb\n" #ram, one less so it can load in gaussian? lol
    f"export GAUSS_MEMDEF={int(mem)-2}gb\n"
    ]
    lines += ["\n"]

    #File Structure
    lines += ["#FILE STRUCUTRE\n"]
    lines += [
    "cd $SLURM_SUBMIT_DIR\n",
    "mkdir -p $SCRATCH/$USER/$SLURM_JOB_ID\n", #Scratch
    "export GAUSS_SCRDIR=$SCRATCH/$USER/$SLURM_JOB_ID\n",
    "\n"
    ]

    #Execution
    lines += ["#EXECUTION AND CLEAN-UP\n"]
    lines += [f"{version} < {name}.com > {name}.out\n"]
    lines += [f"rm -rf $GAUSS_SCRDIR\n"]
    lines += ["\n"]

    #Write pbs lines:
    with open(name + ".pbs",'w+') as file:
        for line in lines:
            file.write(line)

#Submit to queue
if args.debug:
    for filename in args.filenames:
        name = filename.split('.')[0]
        print("Submitting " + name +".pbs" + "...")
        name = filename.split('.')[0]
        print("sbatch " + name + ".pbs")
else:
    for filename in args.filenames:
        name = filename.split('.')[0]
        print("Submitting " + name + ".pbs" + "...")
        name = filename.split('.')[0]
        os.system("sbatch " + name + ".pbs")





