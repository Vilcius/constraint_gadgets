#!/bin/bash
#SBATCH -J generate_vcg_params
#SBATCH -A SIP-UTK0040
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:05:00
#SBATCH -p campus
#SBATCH -e job_files/%j.err
#SBATCH -o job_files/%j.out
#SBATCH --qos=campus

module load anaconda3/2021.05
source $ANACONDA_SH

PROJECT_ROOT="$1"

python3.11 "$PROJECT_ROOT/run/create_vcg_database.py" \
    --generate-params \
    --params-out "$PROJECT_ROOT/run/params/vcg_params.jsonl" \
    --data-dir "$PROJECT_ROOT/data/"
