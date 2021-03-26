#! /bin/bash -l
#
#SBATCH --job-name=inference_pipeline_HPC.sbatch
#SBATCH --open-mode=append
#SBATCH --output=inference_pipeline_HPC.out
#SBATCH --no-requeue
#SBATCH --export=USER,LOGNAME,HOME,MAIL,PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin,JOBGEN_JOINOUTERR=true,JOBGEN_NAME=inference_pipeline_HPC.jgen,JOBGEN_NSLOTS=1,JOBGEN_OUTPATH=inference_template.jgen.out,JOBGEN_PROJECT=nrcan_geobase,JOBGEN_QUEUE=gpu-v100,JOBGEN_SHELL=/bin/bash,JOBGEN_SLOT_IMAGE=nrcan/nrcan_all_default_ubuntu-18.04-amd64_latest,JOBGEN_SLOT_MEMORY=128G,JOBGEN_SLOT_NCORES=8,JOBGEN_SLOT_TMPFS=10G,JOBGEN_WALLCLOCK=6:00:00
#SBATCH --partition=gpu-v100
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16384M
#SBATCH --comment="image=nrcan/nrcan_all_default_ubuntu-18.04-amd64_latest,ssh=true,nsswitch=true"
#SBATCH --gpus-per-task=1

# USER VARIABLES
yaml=path/to/config.yaml

# SET ENVIRONMENT VARIABLES (DO NOT TOUCH)
# load cuda 11.2
. ssmuse-sh -d /fs/ssm/hpco/exp/cuda-11.2.0
export LD_LIBRARY_PATH=/fs/ssm/hpco/exp/cuda-11.2.0/cuda_11.2.0_all/lib64
export PYTHONPATH=/space/partner/nrcan/geobase/work/opt/miniconda3
export PATH=/usr/local/cuda-10.0/bin:$PATH
export MKL_THREADING_LAYER=GNU
source /space/partner/nrcan/geobase/work/opt/miniconda3/bin/activate
# Activate geo-deep-learning conda environment
conda activate /gpfs/fs3/nrcan/nrcan_geobase/work/envs/gdl_py38

# EXECUTION STEPS
# Run inference with desired yaml
cd /gpfs/fs3/nrcan/nrcan_geobase/work/transfer/work/deep_learning/inference/geo-deep-learning
python inference.py -p $yaml

# activate post-processing conda environment.
conda deactivate
conda activate qgis316

# Run post-process.py with directory containing inferences as argument (must contain file ending with '_inference.tif'!)
cd /gpfs/fs3/nrcan/nrcan_geobase/work/transfer/work/deep_learning/inference/
python post-process.py -p $yaml
conda deactivate

# interactive mode
# . ssmuse-sh -x main/opt/jobsubi/jobsubi-0.4
# jobsubi --show-request -R queue=gpu-v100 -R project=nrcan_geobase -r memory=128G -r ncores=8 -r gputype=v100 -r ngpus=1 -r nslots=1 -r w=60 -r wallclock=21600 gpsc3