import argparse
import subprocess
import warnings
from pathlib import Path

from joblib import Parallel, delayed

from utils import read_parameters, get_key_def


def subprocess_command(command: str):
    print(f'Python\'s subprocess executing following command:\n{command}')
    subproc = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = subproc.stdout.decode("utf-8")  # specify encoding
    print(stdout)
    if subproc.stderr:
        warnings.warn(str(subproc.stderr))


def main(img_path, params):
    print(f'Post-processing {img_path}')

    # post-processing parameters
    # FIXME: as yaml input
    classes = [(1, 'forest'), (2, 'hydro'), (3, 'roads'), (4, 'buildings')]
    cell_size_resamp = 0
    orthogonalize_ang_thresh = 20
    to_cog = False
    keep_non_cog = True

    # set name of output gpkg: myinference.tif will become myinference.gpkg
    final_gpkg = Path(img_path).parent / f'{Path(img_path).stem}.gpkg'
    if final_gpkg.is_file():
        warnings.warn(f'Output geopackage exists: {final_gpkg}. Skipping to next inference...')
    else:
        if (len(classes)) != 4:
            raise NotImplementedError
        command = f'qgis_process run model:gdl-{len(classes)}classes -- ' \
                  f'srcinfraster="{img_path}" ' \
                  f'r2vcellsizeresamp={cell_size_resamp} ' \
                  f'native:package_1:dest-gpkg={final_gpkg} '

        # for attrnum, class_name in classes:
        #     if attrnum == 0:
        #         warnings.warn("Are you sure value 0 is of interest? It is usually used to set background class, "
        #                       "i.e. non-relevant class")
        #
        #     command += f'attnum{attrnum}={attrnum} ' \
        #                f'class{attrnum}=\'{class_name}\' '

        subprocess_command(command)

    # COG
    if to_cog:
        # print(f'COGuing {count} of {len(globbed_imgs_paths)}...')
        img_path_cog = img_path.parent / f'{img_path.stem}_cog{img_path.suffix}'
        if img_path_cog.is_file():
            warnings.warn(f'Output cog exists: {str(img_path_cog)}. Skipping to next inference...')
        else:
            cog_command = f'gdal_translate {img_path} {img_path_cog} -co TILED=YES -co COPY_SRC_OVERVIEWS=YES ' \
                          f'-co COMPRESS=LZW'
            subprocess_command(cog_command)
        if keep_non_cog is False and img_path_cog.is_file():
            img_path.unlink(missing_ok=True)


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
    #num_bands = get_key_def('num_bands', params['global'], expected_type=int)
    num_bands = 4
    glob_pattern = f"inference_{num_bands}bands/*_inference.tif"
    globbed_imgs_paths = list(working_folder.glob(glob_pattern))

    print(f"Found {len(globbed_imgs_paths)} inferences to post-process")

    Parallel(n_jobs=len(globbed_imgs_paths))(delayed(main)(file, params=params) for file in globbed_imgs_paths)
    #main(params)