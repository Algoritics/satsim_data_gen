"""
author: Justin Fletcher
date: 20 Jun 2019
"""


import os
import json
import shutil
import argparse
import subprocess
import multiprocessing
import numpy as np
import tensorflow as tf
from itertools import cycle

# I am launched on a node with one param: num_samples


def build_sensor_config(num_samples, num_frames_per_sample, base_config_dict):

    config_dict = base_config_dict

    sim_dict = config_dict['sim']
    sim_dict["samples"] = num_samples
    config_dict['sim'] = sim_dict

    fpa_dict = config_dict['fpa']

    # Select a FOV for this sensor.
    fov = np.random.uniform(0.3, 1.5)
    fpa_dict["y_fov"] = fov
    fpa_dict["x_fov"] = fov

    # Dark current in photoelectrons. Consider Lognormal around about 1.
    fpa_dict["dark_current"] = np.random.uniform(0.5, 5)

    # Gain and bias.
    fpa_dict["gain"] = np.random.uniform(1.0, 2.0)
    fpa_dict["bias"] = np.random.uniform(90, 110)
    fpa_dict["zeropoint"] = np.random.uniform(21.0, 26.0)

    # fpa_dict = config_dict['fpa']
    # a2d_dict = fpa_dict['a2d']
    a2d_dict = dict()
    a2d_dict["response"] = "linear"
    a2d_dict["fwc"] = np.random.uniform(190000, 200000)
    a2d_dict["gain"] = np.random.uniform(1.0, 2.0)
    a2d_dict["bias"] = np.random.uniform(9, 11)
    fpa_dict["a2d"] = a2d_dict

    # Read noise for smae sensor.
    # noise_dict = fpa_dict["noise"]
    noise_dict = dict()
    noise_dict["read"] = np.random.uniform(5, 20)
    noise_dict["electronic"] = np.random.uniform(5, 10)
    fpa_dict["noise"] = noise_dict

    # psf_dict = fpa_dict["psf"]
    psf_dict = dict()
    psf_dict["mode"] = "gaussian"
    psf_dict["eod"] = np.random.uniform(0.05, 0.9)
    fpa_dict["psf"] = psf_dict

    # time_dict = fpa_dict["time"]
    time_dict = dict()
    time_dict["exposure"] = np.random.uniform(1.0, 2.0)
    time_dict["gap"] = np.random.uniform(0.1, 1)
    fpa_dict["time"] = time_dict

    fpa_dict["num_frames"] = num_frames_per_sample

    # Set the FPA.
    config_dict['fpa'] = fpa_dict

    return(config_dict)


def make_clean_dir(directory):
    """Cleans and makes a specified directory tree."""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def mp_worker(cmd_str):
    """Launches a process."""
    print(cmd_str)
    process = subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE)
    process.wait()
    print(process.returncode)


def mp_handler(num_procs, cmd_str_list):
    """Handles mapping processes to pool."""
    pool = multiprocessing.Pool(num_procs)
    pool.map(mp_worker, cmd_str_list)


def main(flags, **kwargs):
    cmd_strings = list()

    # Read the base config file which randomizes over other properties.
    with open(flags.config_file_path, 'r') as f:
        base_config_dict = json.load(f)

    # Generate a new config file randomly selecting from an FPA config.
    for sensor_num, device_num in zip(range(flags.num_sensors), cycle(flags.device)):
        config_dict = build_sensor_config(flags.num_samples,
                                          flags.num_frames_per_sample,
                                          base_config_dict)

        # Create a directory for this sensors runs.
        sensor_name_str = f'sensor_{sensor_num}'
        sensor_dir = os.path.join(flags.output_dir, sensor_name_str)
        # Clear this sensor dir if it exists, then make it.
        make_clean_dir(sensor_dir)

        # Build a filename for this config.
        sensor_json_file = f'sensor_{sensor_num}.json'
        output_config_file = os.path.join(sensor_dir, sensor_json_file)

        # Write a JSON file in the new dir.
        with open(output_config_file, 'w') as fp:
            json.dump(config_dict, fp)

        if flags.debug_satsim:
            cmd_str = f'satsim --debug DEBUG run --device {device_num} ' \
                      + f'--memory {flags.memory} --mode eager --output_dir' \
                      + f' {sensor_dir} --jobs {flags.jobs} {output_config_file}'
        else:
            cmd_str = f'satsim run --device {device_num} --memory ' \
                      + f'{flags.memory} --mode eager --output_dir ' \
                      + f'{sensor_dir} --jobs {flags.jobs} {output_config_file}'

        cmd_strings.append(cmd_str)
        print(cmd_str)
    mp_handler(flags.num_procs, cmd_strings)


if __name__ == '__main__':
    print(tf.__version__)

    # Set arguments and their default values
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_path', type=str,
                        default="satsim.json",
                        help='Path to the JSON config for SatSim.')
    parser.add_argument('--output_dir', type=str,
                        default="output/",
                        help='Path to the JSON config for SatSim.')
    parser.add_argument('--num_sensors', type=int,
                        default=16,
                        help='The number of sensors to simulate.')
    parser.add_argument('--num_samples', type=int,
                        default=8,
                        help='The number of samples from each sensor.')
    parser.add_argument('--num_frames_per_sample', type=int,
                        default=1,
                        help='The number of image frames in each sample.')
    parser.add_argument('--num_frames', type=int,
                        default=6,
                        help='The number of frames to use in each sequence.')
    parser.add_argument('--device', type=int, nargs='+',
                        default=0,
                        help='Number of the GPU(s) to use.')
    parser.add_argument('--debug_satsim', action='store_true',
                        default=False,
                        help='If true, write annotated JPEGs to disk.')
    parser.add_argument('--jobs', type=int,
                        default=1,
                        help='Number of jobs for each sensor per GPU.')
    parser.add_argument('--num_procs', type=int,
                        default=1,
                        help='Number of parallel processes to spawn over all GPUs.')
    parser.add_argument('--memory', type=int,
                        default=7000,
                        help='Max memory use per job in MB. Default = 7000 MB.')

    flags = parser.parse_args()
    main(flags)
