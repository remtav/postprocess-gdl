Author: Rémi Tavon | Date: March 2nd 2021

# *Geo-Deep-Learning*  - Inference & Post-processing Pipeline
> Note: These instruction apply only to linux-based operating systems. Tested on Ubuntu 18.04 and 20.04.

## Configuration

### 1. Inference with geo-deep-learning 
1.1 Clone geo-deep-learning from github repo into desired local directory

   ```.sh
   git clone https://github.com/NRCan/geo-deep-learning/
   ```

1.2 Configure conda environment for geo-deep-learning (see github)
   
    > Note for HPC users: conda environment exists. Skip this step!

### 2. Post-processing with GeoSim*
*\* TEMPORARY: Full integration of post-processing tools on its way.*

2.1 Install GRASS (check with command "grass78 -v" from command line to validate installation) 
    
    > Note for HPC users: GRASS is installed. Simply type following command (or add to .profile for persistence):

`export PATH=/home/ret000/grass-7.8.2/bin.x86_64-pc-linux-gnu/:$PATH`

2.2 Install QGIS*

- Option A: Create a conda environment containing qgis:
conda create --name qgis_316 python=3.8
conda activate qgis_316
conda install qgis=3.16.4 --channel conda-forge

> Note for HPC users: the conda environment exists. Skip step 2.

- Option B: Install QGIS 3.16.4 (long term release, i.e. "ltr") with system's package manager (e.g. apt-get)

2.3 Configure post-processing scripts and models in QGIS

- Option A: Install GeoSim plugin in QGIS 
> (TEMPORARY) Use Option B: full integration of post-processing tools is work in progress.

    - Download GeoSim QGIS plugin from https://github.com/remtav/GeoSim) as .zip
    - Open QGIS, then install QGIS plugin "geo-sim" from .zip

- Option B (TEMPORARY): Extract processing.zip and merge with "/home/[user_name]/.local/share/QGIS/QGIS3/profiles/default/processing"

2.4 (TEMPORARY) Copy "post-process.py" into "path/to/geo-deep-learning"

## Execution
Set paths and run `inference_pipeline.sh` file!

` bash inference_pipeline.sh`