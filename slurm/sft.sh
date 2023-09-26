#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=p-sty-sft
#SBATCH --output=/scratch/bc3088/capstone/dst-capstone/log/%j_%x.out
#SBATCH --error=/scratch/bc3088/capstone/dst-capstone/log/%j_%x.err
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --mem=72GB
#SBATCH --requeue

#SBATCH --mail-type=ALL
#SBATCH --mail-user=bale.chen@nyu.edu

source /scratch/bc3088/env.sh;
conda activate fc3;

bash scripts/lora.sh 4;
