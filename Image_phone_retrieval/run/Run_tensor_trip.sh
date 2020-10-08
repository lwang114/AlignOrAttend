#!/bin/sh
#SBATCH --partition=general --qos=short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu
##SBATCH --nodelist=ewi1
#SBATCH --exclude=insy6,insy12
#SBATCH --workdir=/tudelft.net/staff-bulk/ewi/insy/MMC/xinsheng/code/Interspeech2020/FGSL_V12_F

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.5.1.10
srun -u --output=run/B32.outputs sh run/B32.sh