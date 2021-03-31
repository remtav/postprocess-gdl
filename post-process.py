import argparse
import subprocess
import warnings
from pathlib import Path

from joblib import Parallel, delayed

from utils import read_parameters, get_key_def, load_checkpoint, compare_config_yamls


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
    classes = get_key_def('classes', params['global'], expected_type=dict)

    r2v_cellsize_resamp = get_key_def('r2vect_cellsize_resamp', params['post-processing'], default=0, expected_type=int)
    removeholesunder = get_key_def('removeholesunder', params['post-processing'], default=0, expected_type=int)
    simptol = get_key_def('simptol', params['post-processing'], default=0, expected_type=int)
    redbenddiamtol = get_key_def('redbenddiamtol', params['post-processing'], default=0, expected_type=int)
    recttol = get_key_def('recttol', params['post-processing']['buildings'], default=0, expected_type=int)
    compacttol = get_key_def('compacttol', params['post-processing']['buildings'], default=0, expected_type=int)
    patterntol = get_key_def('patterntol', params['post-processing']['buildings'], default=20, expected_type=int)
    orthogonalize_ang_thresh = get_key_def('orthogonalize_ang_thresh', params['post-processing']['buildings'],
                                           default=0, expected_type=int)

    to_cog = get_key_def('to_cog', params['post-processing'], default=True, expected_type=bool)
    keep_non_cog = get_key_def('keep_non_cog', params['post-processing'], default=True, expected_type=bool)

    # validate inputted classes
    if 0 in classes.keys():
        warnings.warn("Are you sure value 0 is of interest? It is usually used to set background class, "
                      "i.e. non-relevant class. Will add 1 to all class values inputted, e.g. 0,1,2,3 --> 1,2,3,4")
        classes = {cl_val + 1: name for cl_val, name in classes}

    # set name of output gpkg: myinference.tif will become myinference.gpkg
    # FIXME: let user set output directory
    final_gpkg = Path(img_path).parent / f'{Path(img_path).stem}.gpkg'
    if final_gpkg.is_file():
        warnings.warn(f'Output geopackage exists: {final_gpkg}. Skipping to next inference...')
    else:
        if len(classes.keys()) == 1 and classes[1] == 'roads':
            command = f'qgis_process run model:gdl-roads -- ' \
                      f'inputraster="{img_path}" ' \
                      f'r2vcellsizeresamp={r2v_cellsize_resamp} ' \
                      f'native:package_1:dest-gpkg={final_gpkg}'
        elif len(classes.keys()) == 1 and classes[1] == 'buildings':
            command = f'qgis_process run model:gdl-buildings -- ' \
                      f'srcinfraster="{img_path}" ' \
                      f'r2vcellsizeresamp={r2v_cellsize_resamp} ' \
                      f'native:package_1:dest-gpkg={final_gpkg}'
        elif len(classes) == 4:
            command = f'qgis_process run model:gdl-{len(classes)}classes -- ' \
                      f'srcinfraster="{img_path}" ' \
                      f'r2vcellsizeresamp={r2v_cellsize_resamp} ' \
                      f'native:package_1:dest-gpkg={final_gpkg}'
        else:
            raise NotImplementedError(f'Cannot post-process inference with {len(classes.keys())} classes')

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
            try:
                img_path.unlink(missing_ok=True)
            except TypeError:
                img_path.unlink()
            except FileNotFoundError:
                print(f'Could not delete non cog inference: {keep_non_cog}')


if __name__ == '__main__':
    print('\n\nStart:\n\n')
    parser = argparse.ArgumentParser(usage="%(prog)s [-h] [-p YAML] [-i MODEL IMAGE] ",
                                     description='Inference and Benchmark on images using trained model')

    parser.add_argument('-p', '--param', metavar='yaml_file', nargs=1,
                        help='Path to parameters stored in yaml')
    parser.add_argument('-i', '--input', metavar='model_pth img_dir', nargs=2,
                        help='model_path and image_dir')
    args = parser.parse_args()

    # if a yaml is inputted, get those parameters and get model state_dict to overwrite global parameters afterwards
    if args.param:
        input_params = read_parameters(args.param[0])
        model_ckpt = get_key_def('state_dict_path', input_params['inference'], expected_type=str)
        # load checkpoint
        checkpoint = load_checkpoint(model_ckpt)
        if 'params' not in checkpoint.keys():
            warnings.warn('No parameters found in checkpoint. Use GDL version 1.3 or more.')
        else:
            params = checkpoint['params']
            # overwrite with inputted parameters
            compare_config_yamls(yaml1=params, yaml2=input_params, update_yaml1=True)
        del checkpoint
        del input_params

    # elif input is a model checkpoint and an image directory, we'll rely on the yaml saved inside the model (pth.tar)
    elif args.input:
        model_ckpt = Path(args.input[0])
        image = args.input[1]
        # load checkpoint
        checkpoint = load_checkpoint(model_ckpt)
        if 'params' not in checkpoint.keys():
            raise KeyError('No parameters found in checkpoint. Use GDL version 1.3 or more.')
        else:
            # set parameters for inference from those contained in checkpoint.pth.tar
            params = checkpoint['params']
            del checkpoint
        # overwrite with inputted parameters
        params['inference']['state_dict_path'] = args.input[0]
        params['inference']['img_dir_or_csv_file'] = args.input[1]
    else:
        print('use the help [-h] option for correct usage')
        raise SystemExit

    state_dict_path = get_key_def('state_dict_path', params['inference'])
    working_folder = Path(state_dict_path).parent
    #ckpt_num_bands = get_key_def('num_bands', params['global'], expected_type=int)
    #glob_pattern = f"inference_{ckpt_num_bands}bands/*_inference.tif"
    glob_pattern = f"**/*_inference.tif"
    globbed_imgs_paths = list(working_folder.glob(glob_pattern))

    if not globbed_imgs_paths:
        raise FileNotFoundError(f'No tif images found to post-process in {working_folder}')
    else:
        print(f"Found {len(globbed_imgs_paths)} inferences to post-process")

    Parallel(n_jobs=len(globbed_imgs_paths))(delayed(main)(file, params=params) for file in globbed_imgs_paths)
    #main(params)