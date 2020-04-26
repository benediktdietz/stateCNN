"""
Purpose:
    To train models on provided data for the Simulation Images Dataset for specified targets.
    This is used for performing Level 5 Pushing Tasks in which only Pixel/Image data is available for Input.
    The user only has to provide the parameters specified in the `PARAMETERS to be set by User` section.

    CAUTION !!!
        DON'T FORGET TO SET ulimit -n unlimited WITHIN THE SHELL BEFORE STARTING THE TRAINING.
        OTHERWISE ERRORS ARE BOUND TO OCCUR GIVEN THE HIGH NUMBER OF FILES PROVIDED BY THE DATASET STRUCTURE.

Questions:

    matej.zecevic@tuebingen.mpg.de

"""

import os
import torch
from level5_regression.Training import Training
from level5_regression.SimulationImagesDataset import SimulationImagesDataset

###################################################################################################
############### PARAMETERS to be set by User ######################################################

# specify the state space components to predict one
prediction_target = 'ALL' #["angles", "obj_pos", "tip_pos", "target_pos"]  # use None or "ALL" for the full 22-dimensional target

# specify the root directory of the raw dataset
root_dir_dataset = "../development_data/20191022_level5_raw_data_new_generation"  # use the best_policy subset

# if True, then use a concatenated Two-Sequence as Input
sequence_samples = False

# the Training / Validation / Test Set split ratio
# if this is 0.8 then 80% of the data is for Test and rest for Evaluation
ratio = 0.8

# after every <number> epoch, save the model
checkpoint_frequency = 1

# number of epochs to train the model for
num_training_iterations = 200

# to control reproducibility
master_seed = 4444

# path to the PyTorch model of a previous training
# if specified then that model resumes its training
path_checkpoint_file = None

# number of samples per update during training
batch_size = 4

# number of Filters for the first convolutional layer
num_filters = 4

# nonlinearity between layers
activation = "relu"

# set to 0 by default, could possibly have impact on speed
num_workers = 0

# either "xz", "yz", "+yz" or "+yz+xy"
# specifies whether to use the information of a single
# or multiple camera views and which ones
multi_view = "yz"

# if specified, then this is the percantage of a dataset
# that will be used for the actual training
subset_ratio_per_dataset = 0.1  # if None or 1 then the full dataset is being used

# if True, then use a 3-layer Fully Connected regressor
# at the end of the overall CNN architecture
third_fc_layer = False

###################################################################################################
###################################################################################################

torch.manual_seed(master_seed)
print("Torch seeded with Master Seed " + str(master_seed))

# Generate the PyTorch Dataset
simulation_images_dataset = SimulationImagesDataset(root_dir_dataset,
                                                    sequence_samples=sequence_samples,
                                                    prediction_target=prediction_target,
                                                    multi_view=multi_view)


# Compute the Training/Validation/Test Split for the complete Dataset
complete_dataset_length = len(simulation_images_dataset)

training_set_length = int(complete_dataset_length * ratio)

validation_set_length = int(((complete_dataset_length - training_set_length) / 2)) # int(np.round(complete_dataset_length * np.round(((1 - ratio) / 2),1), 1))

test_set_length = validation_set_length
if validation_set_length != ((complete_dataset_length - training_set_length) / 2):
    training_set_length += 1


def get_split(dataset_length, ratio):
    """
    Calculate the split for a dataset given the ratio

    :param dataset_length: (int) length of the dataset
    :param ratio: (int) between 0-1
    :return: (int, int) a split according to ratio, a+b == dataset_length
    """

    a, b = int(round(ratio * dataset_length)), int(round((1 - ratio) * dataset_length))
    assert(a + b == dataset_length)

    return a, b


if subset_ratio_per_dataset:

    training_set_length, training_set_holdout = get_split(training_set_length, subset_ratio_per_dataset)
    validation_set_length, validation_set_holdout = get_split(validation_set_length, subset_ratio_per_dataset)
    test_set_length, test_set_holdout = get_split(test_set_length, subset_ratio_per_dataset)

    training_set, validation_set, test_set, _, _, _ = torch.utils.data.random_split(simulation_images_dataset,
                                                                           [training_set_length,
                                                                            validation_set_length,
                                                                            test_set_length,
                                                                            training_set_holdout,
                                                                            validation_set_holdout,
                                                                            test_set_holdout])

    print("Using only " + str(subset_ratio_per_dataset * 100) + " % of Data Points per Dataset.")

else:

    training_set, validation_set, test_set = torch.utils.data.random_split(simulation_images_dataset,
                                                                           [training_set_length, validation_set_length,
                                                                            test_set_length])

assert(training_set_length == len(training_set)
       and validation_set_length == len(validation_set)
       and test_set_length == len(test_set))

print("Training Set Length: {}"
      "\nValidation Set Length: {}"
      "\nTest Set Length: {}"
      "\nTotal number of Datapoints used for this Training: {}"
      "\nTOTAL NUMBER OF DATAPOINTS IN COMPLETE DATASET: {}"
      .format(str(training_set_length),
              str(validation_set_length),
              str(test_set_length),
              str(training_set_length+validation_set_length+test_set_length),
              str(complete_dataset_length)))


def get_loader(dataset, batch_size=4, num_workers=2):
    """
    Provide the PyTorch loader, which can directly be used for Training.

    :param dataset: (PyTorch.Dataset) dataset instance
    :param batch_size: (int) number of samples to load each iteration during training
    :param num_workers: (int)
    :return: (object) PyTorch Dataset Loader
    """

    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=num_workers) # TODO: inspect whether this variable can better something


# Get Dataset Loaders
loader_training_set = get_loader(training_set, batch_size=batch_size, num_workers=num_workers)

loader_validation_set = get_loader(validation_set, batch_size=batch_size, num_workers=num_workers)

loader_test_set = get_loader(test_set, batch_size=batch_size, num_workers=num_workers)

training = Training(loader_training_set, loader_validation_set, loader_test_set, path_checkpoint_file)

# Set Prediction Target
num_outputs = training.set_prediction_target(prediction_target)

# Start Training
training.train(num_outputs=num_outputs,
               num_filters=num_filters,
               activation=activation,
               num_training_iterations=num_training_iterations,
               checkpoint_frequency=checkpoint_frequency,
               master_seed=master_seed,
               third_fc_layer=third_fc_layer)

# Evaluate on Test Set
test_cost = training.model.validate_model(loader_test_set,
                                                master_seed=training.model.master_seed,
                                                display_random_samples=True,
                                                type="Test Set")
with open(os.path.join(training.model.dir_experiment, "performance_test_loss.log"), "w") as f:
    f.write(str(test_cost.detach().numpy()))
    f.write("\n")