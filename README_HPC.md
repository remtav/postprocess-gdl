Auteur: Rémi Tavon | Date: March 31 2021

# *Geo-Deep-Learning*  - Pipeline d'inférence & post-traitement
> Note: These instruction apply only to HPC.

Répertoire de travail sur HPC pour les inférences:
`/gpfs/fs3/nrcan/nrcan_geobase/work/transfer/work/deep_learning/inference/`

## Configuration

### 1. Inference with geo-deep-learning 
Clone geo-deep-learning from github repo into desired local directory

   ```.sh
   cd /gpfs/fs3/nrcan/nrcan_geobase/work/transfer/work/deep_learning/inference/
   git clone https://github.com/NRCan/geo-deep-learning/
   ```

### 2. Post-processing with GeoSim
2.1 GeoSim

- Download GeoSim QGIS plugin from https://github.com/remtav/GeoSim) as .zip
- Open QGIS, then install QGIS plugin "geo-sim" from .zip
2.2 Postprocess-gdl
  
- Clone postprocess-gdl from github repo into desired local directory

   ```.sh
   cd /gpfs/fs3/nrcan/nrcan_geobase/work/transfer/work/deep_learning/inference/
   git https://github.com/remtav/postprocess-gdl
   ```

## Execution
- Set config in `postprocess-gdl/confif_4class.yaml`
- Set path to config yaml in `postprocess-gdl/inference_pipeline_HPC.sbatch`, then run with `sbatch postprocess-gdl/inference_pipeline_HPC.sbatch` file!