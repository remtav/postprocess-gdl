# Post-processing
import argparse
import subprocess
import warnings
from pathlib import Path
from ruamel_yaml import YAML
import os


def main(params):
    # loop through files from dir
    # working_folder = Path(dir)
    # print(f"Found directory {base_dir}? {base_dir.is_dir()}")
    # glob_pattern = f"*_inference.tif"

    working_folder = Path(params['inference']['state_dict_path']).parent
    glob_pattern = f"inference_?bands/*_inference.tif"
    globbed_imgs_paths = list(working_folder.glob(glob_pattern))
    print(f"Found {len(globbed_imgs_paths)} inferences to post-process")
    # loop through list
    # for img_path in globbed_imgs_paths:
    count = 0
    for img_path in globbed_imgs_paths:
        print(f'COGuing {count} of {len(globbed_imgs_paths)}...')
        img_path_cog = img_path.parent / f'{img_path.stem}_cog{img_path.suffix}'
        cog_command = f'gdal_translate {img_path} {img_path_cog} -of COG -co TILING_SCHEME=GoogleMapsCompatible ' \
                      f'-co COMPRESS=LZW'
        print(cog_command)
        subproc = subprocess.run(cog_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if subproc.stderr:
            warnings.warn(subproc.stderr)

        print(f'Post-processing {count} of {len(globbed_imgs_paths)}...')
        count += 1

        out_gpkg = Path(img_path).parent / f'{Path(img_path).stem}.gpkg'
        if out_gpkg.is_file():
            warnings.warn(f'Output geopackage exists: {str(out_gpkg)}. Skipping to next inference...')
            continue

        command = f'qgis_process run model:gdl-postprocess --inferenceraster="{img_path}" ' \
                  f'native:package_1:dest-gpkg="{str(out_gpkg)}"'

        # logging.debug(f"Trying to pansharp through command-line with following command: {command}")
        print(command)
        subproc = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if subproc.stderr:
            warnings.warn(subproc.stderr)
            # logging.warning(subproc.stderr)
            # logging.warning("Make sure the environment for OTB is initialized. "
            #                "See: https://www.orfeo-toolbox.org/CookBook/Installation.html")


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


if __name__ == '__main__':
    print('\n\nStart:\n\n')
    parser = argparse.ArgumentParser(usage="%(prog)s [-h] [YAML] [-i MODEL IMAGE] ",
                                     description='Inference and Benchmark on images using trained model')

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

    main(params)
