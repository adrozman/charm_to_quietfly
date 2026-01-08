#!/bin/bash -l

#$ -N xfoil_batch # job name: will be displayed by qstat, and is the prefix of the output file
#$ -l h_rt=2:00:00    # how long you are requesting (HH:MM:SS)
#$ -pe omp 4           # request N cores
#$ -m abe              # send email updates on case (at Abort, Begin, Error)
#$ -P turbomac         # project to charge hours to
#$ -j y                # combine error and log file into one file

# SGE_O_WORKDIR is automatically defined containing the directory of the qsub file. This is optional but often useful
cd $SGE_O_WORKDIR

source setup.sh

# Set the CHARM perf file path
#export CHARM_PERF_PATH="/projectnb/turbomac/jerryj/SUI/quietfly/CHARM_runs/TEST4/u-19_100y0_000p19_960r0_000_3500_3500_3500_3500/SUIhovacperf.dat" 

export CHARM_PERF_PATH="inputs/perf.dat"

# Run the pipeline
time {
    python perf_test.py
    python run_pyxfoil.py
    python results.py
}
