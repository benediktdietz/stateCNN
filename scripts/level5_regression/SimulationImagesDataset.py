"""
Purpose:
    A class which can be imported and used for interacting with the Level 5 dataset.
    The dataset class for efficient usage of PyTorch is being defined based on the raw dataset.
    The raw dataset is a directory which contains a sub directory per rollout which then again contains a pkl file
    for every simulation step which is the rendered pixel data and the corresponding 22-dim state description.

Questions:
    matej.zecevic@tuebingen.mpg.de
"""

import glob
import os
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pickle
from PIL import Image

class SimulationImagesDataset(Dataset):
    """Simulation Images dataset which contains the RGB Image/Pixel data for each State."""

    def __init__(self, root_raw_data_directories, transform=None, sequence_samples=False, prediction_target="ALL", multi_view=None):
        """

        :param root_raw_data_directories:
            (string): Top Level Directory which contains all the Rollout directories,
                      which then again contain the .pkl files for every Simulation step
                      ender and corresponding state description
        :param transform:
            (callable, optional): Optional transform to be applied on a sample. Default is specified
        :param sequence_samples:
            (bool, optional): Optional take sequences of two concatenated as Image input data
        :param prediction_target:
            (string or list of string) Provide the State Space Components to be used as Targets.
        :param multi_view:
            (string, optional): Optional consider multiple views of the scene, can be either "xz", "yz", "+yz" or "+yz+xy"
                                Provides different or even more Information for the Input.
        """

        self.multi_view = multi_view

        self.list_raw_data_directories = sorted(glob.glob(os.path.join(root_raw_data_directories, "rollout_*"))) # TODO: this sorted() is not the same as within the OS !

        lengths_within_each_rollout_directory = [len(glob.glob(os.path.join(d, "*"))) for d in
                                                 self.list_raw_data_directories]

        # TODO: remove redundancy here by generalizing
        # make sure that if additional views are being used, that they exist in equal porpotions
        if self.multi_view == "+yz":
            assert(all([l >= 2 for l in lengths_within_each_rollout_directory]))

        elif self.multi_view == "+yz+xy":
            assert(all([l >= 3 for l in lengths_within_each_rollout_directory]))


        self.lengths_rollouts = [0] + [int(d.split("len_")[1].split("_")[0]) for d in self.list_raw_data_directories]
        self.intervals_for_index_search = list(np.cumsum(self.lengths_rollouts))

        # Helper Function: applies the transformations to the image
        def dataset_images_transformation(sample):

            x = sample["input"]

            torch_transformation_scheme = transforms.Compose([
                # not necessary given that the raw data is already squared low resolution # transforms.Resize((100,100)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

            if isinstance(x, np.ndarray):
                x = Image.fromarray(x)

            return {"input": torch_transformation_scheme(x), "target": torch.tensor(sample["target"], dtype=torch.float32)}

        if transform is None:

            self.transform = dataset_images_transformation

            self.transform_mean = 0.5
            self.transform_std = 0.5

        else:

            self.transform = transform # TODO: handle this case better

        self.sequence_samples = sequence_samples

        if self.sequence_samples:

            self.concat_dimension = 2 # this has only been introduced for Debug purposes, keeping it like this should be the way to go
                                      # therefore do not touch this unless you also change the shapes in RegressorCNN/CNN
            print("Concatenating the Input on Dimension " + str(self.concat_dimension))

        self.dict_state_components = {
            "torques": (-22, -19),
            "angles": (-19, -16),
            "velocity": (-16, -13),
            "obj_pos": (-13, -10),
            "obj_rot": (-10, -6),
            "tip_pos": (-6, -3),
            "target_pos": (-3, None),
        }

        self.label_component = prediction_target

        if isinstance(self.label_component, list):

            self.target_subset_index_range = [self.dict_state_components[c] for c in self.label_component]

        elif self.label_component != "ALL" and self.label_component in self.dict_state_components.keys():

            self.target_subset_index_range = self.dict_state_components[self.label_component]

        else:

            self.target_subset_index_range = None

    def load_sample_from_pickle_step_file(self, path_step_file):
        """
        Load a Sample from a given pickle Step file.

        :param path_step_file: (string) path to the Step from Trajectory to be loaded
        :return: (list) the data tuple for the file and possibly the following file
        """

        following_sample = None # by default

        with open(path_step_file, "rb") as f:

            sample = pickle.load(f)

            if self.sequence_samples:

                path_step_file_that_follows = os.path.join(os.path.dirname(path_step_file),
                                                           str(int(os.path.basename(path_step_file).split(".")[0]) + 1) + ".pkl")

                if not os.path.exists(path_step_file_that_follows):

                    return -1, -1

                else:

                    with open(path_step_file_that_follows, "rb") as f:

                        following_sample = pickle.load(f)

        return sample, following_sample

    def process_sample(self, sample, following_sample=None):
        """
        Prepare a Sample for the final Dataset format.

        :param sample:
        :param following_sample:
        :return:
        """

        if self.transform:

            sample = self.transform(sample)

            if self.sequence_samples and following_sample is not None:

                following_sample = self.transform(following_sample)

                sequence_input = torch.cat((sample["input"], following_sample["input"]))

                sequence_input = torch.cat((sequence_input[:3, :], sequence_input[3:, :]), dim=self.concat_dimension)

                import pdb; pdb.set_trace()

                sample = {"input": sequence_input, "target": following_sample["target"]} # using concatenated images, with the following sample target

        if self.target_subset_index_range:

            if isinstance(self.label_component, list):

                target = []
                for r in self.target_subset_index_range:
                    left, right = r
                    target.append(sample["target"][left:right])
                target = torch.tensor(np.hstack(target))

            else:
                left, right = self.target_subset_index_range
                target = sample["target"][left:right]

            sample = {"input": sample["input"], "target": target}

        return sample

    def __len__(self):
        return sum(self.lengths_rollouts)

    def __getitem__(self, idx):
        """
        :param idx: must be smaller then __len__, zero-based index
        :return: the given image, state pair
        """
        assert(idx < self.__len__())

        if torch.is_tensor(idx):
            idx = idx.tolist()

        interval_index = np.searchsorted(self.intervals_for_index_search, idx, side="right")
        rollout_dir_pointed_to_by_idx = self.list_raw_data_directories[interval_index - 1]
        index_within_rollout_dir = idx - self.intervals_for_index_search[interval_index - 1]
        #import pdb; pdb.set_trace()

        # TODO: clean up this redundancy somehow, to allow more general search
        if self.multi_view == "+yz":

            view_dirs = [os.path.join(rollout_dir_pointed_to_by_idx, rollout_dir_pointed_to_by_idx.split(os.path.sep)[-1] + "_view_xz"),
                         os.path.join(rollout_dir_pointed_to_by_idx, rollout_dir_pointed_to_by_idx.split(os.path.sep)[-1] + "_view_yz")]

        elif self.multi_view == "+yz+xy":

            view_dirs = [os.path.join(rollout_dir_pointed_to_by_idx, rollout_dir_pointed_to_by_idx.split(os.path.sep)[-1] + "_view_xz"),
                         os.path.join(rollout_dir_pointed_to_by_idx, rollout_dir_pointed_to_by_idx.split(os.path.sep)[-1] + "_view_yz"),
                         os.path.join(rollout_dir_pointed_to_by_idx, rollout_dir_pointed_to_by_idx.split(os.path.sep)[-1] + "_view_xy")]

        elif self.multi_view == "yz": # just use the yz side view, as alternative to regular xz front view

            view_dirs = [os.path.join(rollout_dir_pointed_to_by_idx, rollout_dir_pointed_to_by_idx.split(os.path.sep)[-1] + "_view_yz")]

        else:

            view_dirs = [os.path.join(rollout_dir_pointed_to_by_idx, rollout_dir_pointed_to_by_idx.split(os.path.sep)[-1] + "_view_xz")]

        list_path_step_files = [os.path.join(view_dir, str(index_within_rollout_dir) + ".pkl") for view_dir in view_dirs]

        samples_from_different_views = []
        for path_step_file in list_path_step_files:

            sample, following_sample = self.load_sample_from_pickle_step_file(path_step_file)

            if sample == -1 and following_sample == -1: # this only happens if we encounter a boundary case

                return self.__getitem__(idx - 1)  # TODO: this can only happen for `sequence_samples=True`,
                                                  #       we just go to one which would not return None

            sample = self.process_sample(sample, following_sample)

            samples_from_different_views.append(sample)

        if len(samples_from_different_views) == 1:

            sample = samples_from_different_views[0]

        elif len(samples_from_different_views) > 1:

            #import pdb; pdb.set_trace()

            sample = {"input": torch.cat([sample["input"] for sample in samples_from_different_views], dim=1),
                      "target": samples_from_different_views[0]["target"]}

        return sample