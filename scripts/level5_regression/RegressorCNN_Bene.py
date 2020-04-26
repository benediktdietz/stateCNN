"""
Purpose:
    A class for the actual CNN architecture in PyTorch (just the layers and non-linearities).
    A class for the actual complete Model/Regressor infrastructure (the loss, optimizer, training loop etc.).
    The latter is used by a final script to perform the training on the Simulation Images Dataset.

Questions:
    matej.zecevic@tuebingen.mpg.de
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import datetime
import time

# Architecture of the CNN i.e., layers and activation functions
class CNN(nn.Module):

    def __init__(self, num_outputs, num_filters=None, activation="relu", num_views=None, sequence_samples=None, third_fc_layer=False):
        """
        Initialize the Neural Network Architecture with PyTorch.

        :param num_outputs: (int) output space dimension/size
        :param num_filters: (int) number of filters for the convolutional layers
        """
        super(CNN, self).__init__()

        self.activation = activation

        if num_filters is None:

            self.num_filters = 8

        else:

            self.num_filters = num_filters

        self.num_outputs = num_outputs

        self.conv1 = nn.Conv2d(3, self.num_filters, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(self.num_filters, int(self.num_filters * 2), 5)

        if num_views is None:

            raise Exception("Could not infer the number of views in the dataset, therefore, cannot infer necessary shapes.")

        elif num_views == 1:
            shape_vertical = 29
        elif num_views == 2:
            shape_vertical = 61
        else:
            shape_vertical = 93

        if sequence_samples is None:

            raise Exception("Could not infer whether sequence samples are used, therefore, cannot infer necessary shapes.")

        elif sequence_samples:
            shape_horizontal = 61
        else:
            shape_horizontal = 29

        self.num_views = num_views
        self.shape_vertical = shape_vertical
        self.shape_horizontal = shape_horizontal

        self.fc1 = nn.Linear(int(self.num_filters * 2) * self.shape_vertical * self.shape_horizontal, 128)  # reshape

        self.third_fc_layer = third_fc_layer
        
        if self.third_fc_layer:

            self.fc3 = nn.Linear(128, 64)
            print("Using a 3-layer FC Regressor at the End of the Network.")

            self.fc2 = nn.Linear(64, self.num_outputs)

        else:
            
            self.fc2 = nn.Linear(128, self.num_outputs)

        self.fc_dropout = nn.Dropout(0.5)

        self.conv_dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        """
        Calculate the Forward Pass of the Network.

        :param x: Data input
        :return: Predicted regression value
        """

        if self.activation == "relu":

            activation = F.relu

        else:

            activation = torch.tanh
            

        x = self.pool(activation(self.conv1(x)))

        x = self.conv_dropout(x)

        x = self.pool(activation(self.conv2(x)))

        x = x.view(-1, int(self.num_filters * 2) * self.shape_vertical * self.shape_horizontal)

        x = activation(self.fc1(x))

        x = self.fc_dropout(x)

        if self.third_fc_layer:

            x = activation(self.fc3(x))

        x = self.fc2(x)

        return x


class RegressorCNN:

    def __init__(self, num_outputs, num_filters=None, activation="relu", master_seed=None, path_checkpoint_file=None,
                 num_views=None, sequence_samples=None, third_fc_layer=False):
        """
        Initialize the overall Model infrastructure.

        :param num_outputs: (int) output space dimension/size
        :param num_filters: (int) number of filters for the convolutional layers
        """

        self.model = CNN(num_outputs, num_filters, activation, num_views, sequence_samples, third_fc_layer)

        self.model_cost = nn.MSELoss()

        self.model_optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        self.path_checkpoint = path_checkpoint_file

        self.continue_from_epoch = None

        if self.path_checkpoint:

            self.load_from_checkpoint()

        self.dir_experiment = None

        self.master_seed = master_seed

    def load_from_checkpoint(self):
        """
        Load the Model Parameters from a corresponding Checkpoint file.
        """

        checkpoint = torch.load(self.path_checkpoint)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model_optimizer.load_state_dict(checkpoint['model_optimizer_state_dict'])

        self.continue_from_epoch = checkpoint["epoch"]

        print("Continuing training of existing Model from Epoch " + str(self.continue_from_epoch) + " on.")

    def validate_model(self, dataset_loader, master_seed=None, display_random_samples=True, model_batch_size=4, type="Data Set"):
        """
        Validate the performance of the Model on some Dataset.

        :param dataset_loader:
            (object) PyTorch dataset loader to provide samples for evaluation
        :param display_random_samples:
            (bool) if True, then show concrete samples with Ground Truth and Prediction
        :return: (float) the overall cost on that dataset
        """

        if master_seed is None:

            master_seed = 0
            print("'Master' Seed set to " + str(master_seed))

        test_cost = 0

        n_examples = 5

        np.random.seed(master_seed)

        random_indices = np.random.randint(0, len(dataset_loader), n_examples)  # show five random examples

        print("Showing Random Examples Indices: " + str(sorted(list(random_indices))))

        with torch.no_grad():  # IMPORTANT !!! to avoid memory issues, operations inside don't track history

            for i, data in enumerate(dataset_loader, 0):

                test_inputs, test_labels = (data["input"], data["target"])

                predictions_test_batch = self.model(test_inputs)

                if display_random_samples and i in random_indices:

                    for ind, d in enumerate(predictions_test_batch):

                        np.random.seed(master_seed)

                        random_index_in_batch = np.random.randint(0, model_batch_size) # TODO: remove the model_batch_size parameter, as it is not logically necessary

                        if ind == random_index_in_batch:

                            print("Prediction " + str(ind) + ": " + str(d.tolist()) +
                                  "\nGround Truth " + str(ind) + ": " + str(test_labels[ind].tolist()) + "\n")

                test_cost += self.model_cost(predictions_test_batch, test_labels)

        print("[" + str(type) + "] Overall Loss " + str(test_cost.item()) + " (Average Cost per Batch: " + str(test_cost.item() / len(dataset_loader)) + ")")

        return test_cost



    def train_model(self, num_training_iterations, training_set_loader, prediction_target_name, checkpoint_frequency=10, validation_set_loader=None):
        """
        Train the Regressor.

        :param num_training_iterations: (int) number of epochs to train for
        :param checkpoint_frequency: (int) number of epochs after which the model is always saved
        :param validation_set_loader: (object) PyTorch dataset loader to provide samples for evaluation
        """

        print("Checkpoint Save Frequency set to: " + str(checkpoint_frequency))

        print("Starting training of the Model for " + str(num_training_iterations) + " Epochs.")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M_")

        if self.path_checkpoint is not None:

            self.dir_experiment = os.path.dirname(self.path_checkpoint)

        else:

            self.dir_experiment = timestamp + prediction_target_name + "_level5_CNN_model/"

        name_performance_epoch = "performance_epoch_loss.log"
        name_performance_validation = "performance_validation_loss.log"

        time_start_training = time.time()

        list_loss_training = []

        last_epoch = 1

        if self.continue_from_epoch is not None:

            epoch_start = self.continue_from_epoch

            last_epoch = self.continue_from_epoch

        else:

            epoch_start = 1

        for epoch in range(epoch_start,
                           epoch_start + num_training_iterations + 1):  # loop over the dataset multiple times

            # if we only wish to validate, or actually reach the end of the training
            if num_training_iterations == 0 or epoch == (epoch_start + num_training_iterations):
                break

            epoch_loss = 0.0
            running_loss = 0.0
            for i, data in enumerate(training_set_loader, 0):

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = (data["input"], data["target"])

                # zero the parameter gradients
                self.model_optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)

                cost = self.model_cost(outputs, labels)

                cost.backward()

                self.model_optimizer.step()

                # print statistics
                running_loss += cost.item()

                if i % 10 == 9:  # print every 2000 mini-batches

                    if last_epoch != epoch:

                        mode_flush = False

                    else:

                        mode_flush = True

                    print("[Epoch %d, Batch %5d] Cost: %.6f" %
                          (epoch, i + 1, running_loss / 10), end="\r", flush=mode_flush)

                    epoch_loss += running_loss

                    running_loss = 0.0

                    last_epoch = epoch

            print("[Epoch %d] Overall Loss: %.6f                     ---" % (epoch, epoch_loss))

            list_loss_training.append(epoch_loss)

            if not os.path.exists(self.dir_experiment):

                os.makedirs(self.dir_experiment)

            if epoch % checkpoint_frequency == 0:

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'model_optimizer_state_dict': self.model_optimizer.state_dict(),
                    'model_init_params_dict': {
                        "num_outputs": self.model.num_outputs,
                        "num_filters": self.model.num_filters,
                        "activation": self.model.activation,
                        "num_views": self.model.num_views,
                        "sequence_samples": self.model.sequence_samples,
                        "third_fc_layer": self.model.third_fc_layer,
                    },
                    'epoch_loss': epoch_loss,
                },
                    os.path.join(self.dir_experiment, timestamp + "checkpoint_epoch_" + str(epoch) + ".th"))

                print("Saved Checkpoint.")

                if validation_set_loader:

                    validation_cost = self.validate_model(validation_set_loader, master_seed=self.master_seed,
                                                          display_random_samples=True, type="Validation Set")

                    if not os.path.exists(os.path.join(self.dir_experiment, name_performance_validation)):

                        attribute_open = "w"

                    else:

                        attribute_open = "a"

                    # just a clone of below, but to also keep the "validation" loss (which is test loss, as we don't have a proper keep out validation set)
                    with open(os.path.join(self.dir_experiment, name_performance_validation), attribute_open) as f:

                        f.write(str(validation_cost.detach().numpy()))
                        f.write("\n")

            # write progress to log file
            if not os.path.exists(os.path.join(self.dir_experiment, name_performance_epoch)):

                attribute_open = "w"

            else:

                attribute_open = "a"

            with open(os.path.join(self.dir_experiment, name_performance_epoch), attribute_open) as f:

                f.write(str(epoch_loss))
                f.write("\n")

        time_total_training = (time.time() - time_start_training) / 60
        print("Finished Training Model within " + str(time_total_training) + " Minutes.")
