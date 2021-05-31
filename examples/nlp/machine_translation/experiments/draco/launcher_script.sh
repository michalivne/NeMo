#!/usr/bin/env bash
# Usage:
# ./auto_launcher.sh -n 5 <file.sub>

# Grab command line options
# n: Number of times to submit the job
N_CALLS=1
while getopts "n:J:" opt; do
  case $opt in
    n) N_CALLS=$OPTARG;;
  esac
done

# Grab the .sub file to run
SUBFILE=${@:$OPTIND:1}
if [[ -z $SUBFILE ]]; then
  echo "Usage: $(basename "$0") [flags] [sub file]"
  exit 1
fi

echo "Calling [$SUBFILE] $N_CALLS times."

# Repeat calls
#for i in {0..$N_CALLS}
for (( i = 1; i <= $N_CALLS; i++ ))
do
  echo "Submitting job ${i}"
  OUTPUT=$(sbatch $SUBFILE)
  JOBID="$(cut -d' ' -f4 <<< $OUTPUT)"
  while (squeue | grep $JOBID ); do
    sleep 15m
  done
done
