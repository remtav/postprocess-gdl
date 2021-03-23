# Post-processing
import argparse
import subprocess
import uuid
import warnings
from pathlib import Path
from ruamel_yaml import YAML

from joblib import Parallel, delayed


def read_parameters(param_file):
    """Read and return parameters in .yaml file
    Args:
        param_file: Full file path of the parameters file
    Returns:
        YAML (Ruamel) CommentedMap dict-like object
    """
    yaml = YAML()
    with open(param_file) as yamlfile:
        params = yaml.load(yamlfile)
    return params


def subprocess_command(command: str):
    print(f'Python\'s subprocess executing following command:\n{command}')
    subproc = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = subproc.stdout.decode("utf-8")  # specify encoding
    print(stdout)
    if subproc.stderr:
        warnings.warn(str(subproc.stderr))


def main(img_path):
    print(f'Post-processing {img_path}')

    # set name and create directory for temporary gpkg
    temp_gpkg_dir = Path('/tmp') / f'gdl_qgis_process-{uuid.uuid4().hex[0:6]}'
    temp_gpkg_dir.mkdir(exist_ok=True, parents=True)
    temp_gpkg = temp_gpkg_dir / f'{uuid.uuid4().hex[0:6]}.gpkg'

    # set name of output gpkg
    classes = {0: 'forest', 1: 'vegetation', 2: 'roads', 3: 'buildings'}
    classes_output = {}
    for k, v in classes.items():
        classes_output[k] = {'name': v,
                             'output_raw': Path(img_path).parent / f"{v}_raw.gpkg",
                             'output_final': Path(img_path).parent / f"{v}_final.gpkg"}

    final_gpkg = Path(img_path).parent / f'{Path(img_path).stem}.gpkg'
    if final_gpkg.is_file():
        warnings.warn(f'Output geopackage exists: {final_gpkg}. Skipping to next inference...')
    else:
        cell_size_resamp = 0  # FIXME: as input
        # Raster to vector (polygonization)
        command_r2v = f'qgis_process run model:r2vect ' \
                  f'-- inputraster="{img_path}" ' \
                  f'cellsizeresamp={cell_size_resamp} ' \
                  f'grass7:r.to.vect_1:r2vect_output="{temp_gpkg}"'

        subprocess_command(command_r2v)

        for cl, info in classes_output.items():
            if info["name"] == 'roads':
                # Simplify and squeletonize roads
                command = f'qgis_process run model:simplify-road -- ' \
                          f'rtovectoutput="{temp_gpkg}" ' \
                          f'attrnum={cl + 1} ' \
                          f'classname="{info["name"]}" ' \
                          f'reducebenddiamtol=3 ' \
                          f'GeoSimplification:chordalaxis_1:final_gpkg="{info["output_final"]}" ' \
                          f'native:fixgeometries_4:raw_gpkg="{info["output_raw"]}"'
            elif info["name"] == 'buildings':
                # Simplify buildings
                command = f'qgis_process run model:simplify-buildings -- ' \
                          f'rtovectoutput="{temp_gpkg}" ' \
                          f'attrnum={cl+1} ' \
                          f'classname="{info["name"]}" ' \
                          f'reducebenddiamtol=3 ' \
                          f'orthomaxtol=20 ' \
                          f'native:fixgeometries_4:raw_gpkg="{info["output_raw"]}"' \
                          f'native:orthogonalize_1:final_gpkg="{info["output_final"]}"'
            else:
                # Simplify natural classes (hydro, forest)
                command = f'qgis_process run model:simplify-per-class -- ' \
                          f'rtovectoutput="{temp_gpkg}" ' \
                          f'attrnum={cl+1} ' \
                          f'classname="{info["name"]}" ' \
                          f'reducebenddiamtol=3 ' \
                          f'native:fixgeometries_4:raw_gpkg="{info["output_raw"]}" ' \
                          f'GeoSimplification:reducebend_1:final_gpkg={info["output_final"]}'
            subprocess_command(command)

            #pkg_command = f'qgis_process run native:package ' \
                           #f'-- LAYERS=' \

            # pkg_command = f'qgis_process run model:package4 -- ' \
            #               f'layer1="{classes_output[0]["output_raw"]}|layername={classes_output[0]["name"]}_raw" ' \
            #               f'layer2="{classes_output[0]["output"]}|layername={classes_output[0]["name"]}_final" ' \
            #               f'layer3="{classes_output[1]["output"]}|layername={classes_output[1]["name"]}_raw" ' \
            #               f'layer4="{classes_output[1]["output"]}|layername={classes_output[1]["name"]}_final" ' \
            #               f'layer5="{classes_output[2]["output"]}|layername={classes_output[2]["name"]}_raw" ' \
            #               f'layer6="{classes_output[2]["output"]}|layername={classes_output[2]["name"]}_final" ' \
            #               f'layer7="{classes_output[3]["output"]}|layername={classes_output[3]["name"]}_raw" ' \
            #               f'layer8="{classes_output[3]["output"]}|layername={classes_output[3]["name"]}_final" ' \
            #               f'native:package_1:dest_gpkg="{final_gpkg}"'
            # subprocess_command(pkg_command)

    # for cl in classes_output.keys():
    #     cl["output_raw"].unlink()
    #     cl["output_final"].unlink()
    # temp_gpkg.unlink()

    # COG
    # print(f'COGuing {count} of {len(globbed_imgs_paths)}...')
    img_path_cog = img_path.parent / f'{img_path.stem}_cog{img_path.suffix}'
    if img_path_cog.is_file():
        warnings.warn(f'Output cog exists: {str(img_path_cog)}. Skipping to next inference...')
    else:
        cog_command = f'gdal_translate {img_path} {img_path_cog} -co TILED=YES -co COPY_SRC_OVERVIEWS=YES ' \
                      f'-co COMPRESS=LZW'
        subprocess_command(cog_command)

if __name__ == '__main__':
    print('\n\nStart:\n\n')
    parser = argparse.ArgumentParser(usage="%(prog)s [-h] [YAML] [-i MODEL IMAGE] ",
                                     description='Post-processing of inference created by geo-deep-learning')

    parser.add_argument('-p', '--param', metavar='yaml_file', nargs=1,
                        help='Path to parameters stored in yaml')
    parser.add_argument('-i', '--input', metavar='model_pth', nargs=1,
                        help='model_path')
    args = parser.parse_args()

    if args.param:
        params = read_parameters(args.param_file)
    elif args.input:
        model = Path(args.input[0])

        params = {'inference': {}}
        params['inference']['state_dict_path'] = args.input[0]

    else:
        print('use the help [-h] option for correct usage')
        raise SystemExit

    working_folder = Path(params['inference']['state_dict_path']).parent
    glob_pattern = f"inference_?bands/*_inference.tif"
    globbed_imgs_paths = list(working_folder.glob(glob_pattern))

    print(f"Found {len(globbed_imgs_paths)} inferences to post-process")

    Parallel(n_jobs=len(globbed_imgs_paths))(delayed(main)(file) for file in globbed_imgs_paths)
    #main(params)

# qgis_process help model:r2vect
# ----------------
# Arguments
# ----------------
#
# cellsizeresamp: cell_size_resamp
# 	Argument type:	number
# 	Acceptable values:
# 		- A numeric value
# inputraster: input_raster
# 	Argument type:	raster
# 	Acceptable values:
# 		- Path to a raster layer
# grass7:r.to.vect_1:r2vect_output: r2vect_output
# 	Argument type:	vectorDestination
# 	Acceptable values:
# 		- Path for new vector layer
#
# ----------------
# Outputs
# ----------------
#
# grass7:r.to.vect_1:r2vect_output: <outputVector>
# 	r2vect_output

# qgis_process help model:simplify-buildings
# simplify-buildings (model:simplify-buildings)
#
# ----------------
# Description
# ----------------
#
# ----------------
# Arguments
# ----------------
#
# attrnum: attr_num
# 	Argument type:	number
# 	Acceptable values:
# 		- A numeric value
# classname: class_name
# 	Argument type:	string
# 	Acceptable values:
# 		- String value
# orthomaxtol: ortho_max_tol
# 	Argument type:	number
# 	Acceptable values:
# 		- A numeric value
# reducebenddiamtol: red_bend_diam_tol
# 	Argument type:	number
# 	Acceptable values:
# 		- A numeric value
# rtovectoutput: r.to.vect output
# 	Argument type:	vector
# 	Acceptable values:
# 		- Path to a vector layer
# native:package_1:dest-gpkg: dest-gpkg
# 	Argument type:	fileDestination
# 	Acceptable values:
# 		- Path for new file
#
# ----------------
# Outputs
# ----------------
#
# native:package_1:dest-gpkg: <outputFile>
# 	dest-gpkg
#
# qgis_process help model:simplify-per-class
# simplify-per-class (model:simplify-per-class)
#
# ----------------
# Description
# ----------------
#
# ----------------
# Arguments
# ----------------
#
# attrnum: attr_num
# 	Argument type:	number
# 	Acceptable values:
# 		- A numeric value
# classname: class_name
# 	Argument type:	string
# 	Acceptable values:
# 		- String value
# reducebenddiamtol: red_bend_diam_tol
# 	Argument type:	number
# 	Acceptable values:
# 		- A numeric value
# rtovectoutput: r.to.vect output
# 	Argument type:	vector
# 	Acceptable values:
# 		- Path to a vector layer
# native:package_1:dest-gpkg: dest-gpkg
# 	Argument type:	fileDestination
# 	Acceptable values:
# 		- Path for new file
#
# ----------------
# Outputs
# ----------------
#
# native:package_1:dest-gpkg: <outputFile>
# 	dest-gpkg
