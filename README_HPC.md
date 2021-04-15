Auteur: Rémi Tavon | Date: 15 avril 2021

# *Geo-Deep-Learning*  - Pipeline d'inférence & post-traitement
> Note: Ces instructions s'appliquent uniquement à une utilisation sur HPC.

>Répertoire de travail sur HPC pour les inférences:
`/gpfs/fs3/nrcan/nrcan_geobase/work/transfer/work/deep_learning/inference/`

## Étape 1. Inférence avec geo-deep-learning 

### 1.1 Lister les images dans un fichier .csv

Dans Excel, préparer un tableau où chaque rangée contient le chemin absolu vers les images sur lesquelles vous voulez inférer.

Si la vérité terrain est disponible, le script d'inférence pourra alors calculer des métriques en comparant l'inférence à celle-ci. Pour utiliser les annotations en question, ajouter le chemin vers le Geopackage dans la troisième colonne.

Par exemple:

image | (colonne vide) | annotation (optionel)
--- | --- | ---
/gpfs/ [...] /GDL_all_images/image1.tif |  | /gpfs/ [...] /geopackage/image1.gpkg
/gpfs/ [...] /GDL_all_images/image2.tif |  | /gpfs/ [...] /geopackage/image2.gpkg
... |  |...

Exporter sous la forme d'un fichier csv (séparateur: virgule) 

> Le .csv peut également être créé de toutes pièces avec un éditeur de texte de votre choix plutôt qu'avec Excel.

Utiliser FileZilla (ou logicial équivalent) pour transférer ce .csv de votre poste vers le répertoire de travail sur HPC (voir chemin ci-dessus).

### 1.2 Choisir son modèle

Le dossier `./models` contient les meilleurs modèles à jour, classés en fonction du type de données sur lesquels ils ont été entraînés (ex.: SAR ou imagerie satellite). Par exemple, le modèle pour l'extraction des 4 classes fondamentales à utiliser présentement se trouve ici:
`./models/sat-imagery-optical-50cm/4class.pth.tar`. À titre d'information, il s'agit d'une architecture Deeplabv3, entraîné sur l'imagerie RGB-NIR (4 bandes) à 50 cm.

> Peu de modèles ont été placés ici pour l'instant. S'en remettre aux développeurs pour de l'assistance avec de nouveaux modèles qui n'y seraient pas.

### 1.3 Préparer le fichier de configuration .yaml

Le dossier postprocess-gdl contient trois fichiers .yaml qui servent de gabarit en fonction de la tâche d'extraction à effectuer.

`config_4class.yaml` : extraction des 4 classes fondamentales (forêts, hydro, routes, bâtiments)

`config_roads.yaml` : extraction des routes seulement

`config_buildings.yaml` : extraction des bâtiments seulement

Selon vos besoins, copier localement un des fichiers de configuration, puis ouvrir ce fichier dans un éditeur de texte de votre choix.
D'autres fichiers de configuration peuvent être créés en fonction des besoins, bien sûr.

**Section `global`**

```yaml
global:
  task: 'segmentation'
  number_of_bands: 4  # Number of bands in input imagery
  # Set to True if the first three channels of imagery are blue, green, red. Set to False if Red, Green, Blue
  classes:
    1: 'forests'
    2: 'hydro'
    3: 'roads'
    4: 'buildings'
  BGR_to_RGB: True
```

Les trois premiers paramètres `task`, `number_of_bands` et `classes` sont uniquement utilisés à des fins de validation:
- est-ce que le modèle est effectivement compatible pour la tâche à effectuer ? Par exemple, est-ce que le modèle extrait bel et bien 4 classes?
- est-ce que l'imagerie contient le bon nombre de bandes pour ce modèle?
- est-ce que les annotations, si elles sont fournies, contiennent les classes souhaitées pour une comparaison adéquate avec les prédictions?

Le paramètre `BGR_to_RGB` est un booléen qui réorganise les trois premières bandes afin qu'elles soient Rouge-Vert-Bleu plutôt que Bleu-Vert-Rouge. Ainsi, si les trois premières bandes de l'imagerie sont Bleu-Vert-Rouge, il est important de spécifier `BGR_to_RGB: True`. 
>Ce paramètre n'agit que sur les trois premières bandes et est utilisé même si l'imagerie contient plus de trois bandes. Des problèmes pourraient survenir si l'imagerie contient une seule bande. À tester. 

**Section `inference`**

```yaml
# Inference parameters; used in inference.py --------
inference:
  img_dir_or_csv_file: #/path/to/img_dir_or_csv_file
  state_dict_path: # /path/to/model/weights/for/inference/checkpoint.pth.tar
```

Ces paramètres sont utilisés au coeur du script d'inférence.

`img_dir_or_csv_file` : chemin (absolu, de préférence) vers le .csv créé (voir étape 1.1). Ce paramètre accepte également un chemin vers un dossier, dans lequel le script d'inférence cherche automatiquement des images. Cette seconde approche n'est toutefois pas recommandée. Certaines images peuvent être ignorées si elles n'ont pas l'extention .tif ou .TIF, par exemple.

`state_dict_path` : chemin (absolu, de préférence) vers le modèle entraîné qui doit être utilisé pour l'inférence. L'extension de ce modèle est généralement `.pth.tar`.

### 1.4 Exécuter l'inférence via le fichier `.sbatch`

Avant d'indiquer le lien entre le fichier `.yaml` et `.sbatch`, présentons rapidement la fonction de ce troisième fichier et dernier fichier (1 - `.csv`, 2 - `.yaml`, 3 - `.sbatch`).

Le fichier `.sbatch` est utilisé par le gestionnaire de tâche sur HPC pour configurer une machine virtuelle qui servira à effectuer l'inférence. 
Voici quelques notes à propos de son contenu. La première section (qui contient tous les paramètres `#SBATCH`) sert à "créer" une machine temporaire. Toute cette section peut normalement être laissée telle quelle.

```shell
#! /bin/bash -l
#
#SBATCH --job-name=inference_pipeline_HPC.sbatch
#SBATCH --open-mode=append
#SBATCH --output=inference_pipeline_HPC.out
```
Le paramètre `--output` spécifie le nom et l'emplacement du fichier de log qui sera créé pendant l'exécution. Une valeur possible serait, par exemple, `#SBATCH --output=/home/ret000/projet-en-cours.out`

```shell
#SBATCH --no-requeue
#SBATCH --export=USER,LOGNAME,HOME,MAIL,PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin,JOBGEN_JOINOUTERR=true,JOBGEN_NAME=inference_pipeline_HPC.jgen,JOBGEN_NSLOTS=1,JOBGEN_OUTPATH=inference_template.jgen.out,JOBGEN_PROJECT=nrcan_geobase,JOBGEN_QUEUE=gpu-v100,JOBGEN_SHELL=/bin/bash,JOBGEN_SLOT_IMAGE=nrcan/nrcan_all_default_ubuntu-18.04-amd64_latest,JOBGEN_SLOT_MEMORY=128G,JOBGEN_SLOT_NCORES=8,JOBGEN_SLOT_TMPFS=10G,JOBGEN_WALLCLOCK=6:00:00
#SBATCH --partition=gpu-v100
#SBATCH --time=6:00:00
```
Le paramètre `--time` indique le temps maximal alloué pour la tâche. Ce paramètre varie donc en fonction du nombre d'images sur lesquelles inférer et en fonction de la taille du modèle (nombre de paramètres).

```shell
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16384M
#SBATCH --comment="image=nrcan/nrcan_all_default_ubuntu-18.04-amd64_latest,ssh=true,nsswitch=true"
#SBATCH --gpus-per-task=1
```

```shell
# USER VARIABLES
yaml=absolute/path/to/config.yaml
```
**C'est ici qu'il faut spécifier le chemin vers le `.yaml`. Il s'agit de la seule ligne qu'il faudra obligatoirement modifier pour chaque nouvelle tâche d'inférence.** À noter que le même `.yaml` peut être modifié pour différentes tâches qui se succèdent dans le temps. Si le nom demeure inchangé, alors il n'y aurait pas besoin d'éditer le fichier `.batch`.
```shell
# SET ENVIRONMENT VARIABLES (DO NOT TOUCH)
```
Toutes les autres lignes du `.sbatch` (suivant le "DO NOT TOUCH") **ne devraient pas être modifiées**. C'est une section réservée aux développeurs.

### 1.5 Récupérer les inférences

À mesure qu'elles sont produites par le modèle, les inférences brutes, en format raster, pourront être récupérées dans le dossier qui contient le modèle `.pth.tar` que vous avez spécifié dans le `.yaml`. Par exemple:
```
├── inference
    └── models
        └── sat-imagery-optical-50cm
            └── 4class.pth.tar
            └── inference_[nombre de bandes]_bands
                └── *** INFÉRENCES ICI! ***
    └── postprocess-gdl
            └── config_4class.yaml
            └── inference_pipieline_HPC.sbatch
            └── ...
    └── geo-deep-learning
            └── ...
    └── ...
```

Transférer les inférences localement via FileZilla. Voilà qui complète la première étape.

## Étape 2. Post-traitement avec QGIS
> Note: Cette étape devra être effectuée localement pour l'instant. Qgis_process n'est pas encore pleinement fonctionnel sur HPC

### 2.1 Geo Simplification (QGIS)

- Dans QGIS (v3.14+), sous l'onglet "Extensions", cliquer sur "Installer/Gérer les extensions".
- Chercher et installer "Geo Simplification (processing)"

Cette extension contient les algorithmes utiles pour la simplification et pour extraire des lignes à partir des polygones de routes. Une version expériementale permet aussi de faire du post-traitement ciblé pour les bâtiments. Voir avec les développeurs pour cet outil (Daniel Pilon!).

### 2.2 Utiliser le modeleur graphique de QGIS

Les modèles préconstruits pour le post-traitement peuvent être récupérés ìci:
https://github.com/remtav/postprocess-gdl

Dans github, il est possible de télécharger ce répertoire comme .zip (Bouton vert "Code" --> "Download ZIP"). Puis, dans QGIS, les fichiers .model3 peuvent être importés (Traitement --> Boîte à outils --> Ajouter un modèle à la boîte à outils)

Avant d'utiliser un modèle pour une première fois, il vaut mieux l'ouvrir ("Traitement" --> "Modeleur graphique") pour s'assurer que les références aux différents algorithmes sont valides. Selon vos préférences, il peut être pertinent d'exposer davantage de paramètres aux utilisateurs.