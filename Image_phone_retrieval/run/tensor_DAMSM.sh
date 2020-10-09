#!/bin/sh
#SBATCH --partition=general --qos=long
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:turing
##SBATCH --nodelist=ewi1
#SBATCH --exclude=insy6,insy12
#SBATCH --chdir=/tudelft.net/staff-bulk/ewi/insy/MMC/xinsheng/code/Liming/DAVEnet_for_retrieval

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.5.1.10
srun -u --output=run/tensor_DAMSM.outputs python run_phone.py --resume True --losstype DAMSM --feature tensor --data_dir /tudelft.net/staff-bulk/ewi/insy/MMC/xinsheng/data/coco

                        
                               