
from typing import Sequence
import matplotlib.image as mpimg
import numpy as np

from pytest import fixture
import pdb
from moviepy.editor import VideoFileClip

# enable import from modules lying in parent directory
from inspect import getsourcefile
import os

import sys

current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]

sys.path.insert(0, parent_dir)

from find_lanes import *

# CALL THIS VIA PYTEST FROM INSIDE OF "tests" FOLDER!!

# PATHS

@fixture()
def paths_images() -> Sequence[str]:
    dir_images = "../test_images"
    filenames_images = os.listdir(dir_images)
    fullpaths = [os.path.join(dir_images, filename) for filename in filenames_images]
    return fullpaths

@fixture
def paths_images_output(paths_images: Sequence[str]) -> Sequence[str]:
    dir_output_images = "../test_images_output"
    input_filenames = [os.path.split(fullpath)[1] for fullpath in paths_images]
    output_filenames =  [os.path.join(dir_output_images, filename) for filename in input_filenames]
    return output_filenames

@fixture()
def paths_videos() -> Sequence[str]:
    dir_videos = "../test_videos"
    filenames_videos = os.listdir(dir_videos)
    fullpaths = [os.path.join(dir_videos, filename) for filename in filenames_videos]
    return fullpaths

@fixture
def paths_videos_output(paths_videos: Sequence[str]) -> Sequence[str]:
    dir_output_videos = "../test_videos_output"
    input_filenames = [os.path.split(fullpath)[1] for fullpath in paths_videos]
    output_filenames =  [os.path.join(dir_output_videos, filename) for filename in input_filenames]
    return output_filenames



# Does it process single images
def test_process_all_test_images(paths_images: Sequence[str], paths_images_output: Sequence[str]) \
    -> Sequence[np.ndarray]:
    expected_file_saved = True

    i = 0
    output_paths = paths_images_output
    for path in paths_images:
        assert find_lanes_in_images(path, output_paths[i]) == expected_file_saved
        i = i + 1 
        

# Does it process videos
def test_process_all_test_videos(paths_videos: Sequence[str], paths_videos_output: Sequence[str]) \
    -> Sequence[np.ndarray]:
    expected_file_saved = True
    i = 0
    output_paths = paths_videos_output
    for path in paths_videos:
        assert find_lanes_in_videos(path, output_paths[i])  == expected_file_saved
        i = i + 1