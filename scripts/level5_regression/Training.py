"""
Purpose:
    A class for the Training of the CNN regressor.
    This class and its functions are eventually called by a script.

Questions:
    matej.zecevic@tuebingen.mpg.de
"""
from .RegressorCNN import RegressorCNN

class Training:

    def __init__(self, loader_training_set, loader_validation_set=None, loader_test_set=None, path_checkpoint_file=None):
        """
        Initialize a Training Process.

        :param loader_training_set: (object) PyTorch dataset loader to provide samples for training
        :param loader_validation_set: (object) PyTorch dataset loader to provide samples for evaluation
        :param loader_test_set: (object) PyTorch dataset loader to provide samples for evaluation
        :param path_checkpoint_file: (string) path to a Model for which the training should be resumed
        """

        self.loader_training_set = loader_training_set

        self.loader_validation_set = loader_validation_set

        self.loader_test_set = loader_test_set

        self.path_checkpoint_file = path_checkpoint_file

        self.dict_state_components = {
             "torques": (-22,-19),
             "angles": (-19,-16),
             "velocity": (-16,-13),
             "obj_pos": (-13,-10),
             "obj_rot": (-10,-6),
             "tip_pos": (-6,-3),
             "target_pos": (-3,None),
        }

        self.prediction_target = None


    def set_prediction_target(self, prediction_target):
        """
        Get the size of the regression output.

        :param prediction_target:
            (string or list of string) all state space components to be used for Targets
        :return: (int) number of dimensions for the chosen prediction target(s)
        """

        self.prediction_target = prediction_target

        def get_num(p):
            print(">>> PREDICTION TARGET is: " + str(p))
            if self.prediction_target == "obj_rot":
                num_outputs = 4
            elif self.prediction_target == "ALL":
                num_outputs = 22
            else:
                num_outputs = 3
            return num_outputs

        num_outputs = 0
        if isinstance(self.prediction_target, list):
            for p in prediction_target:
                num_outputs += get_num(p)
        else:
            num_outputs = get_num(self.prediction_target)

        return num_outputs


    def train(self, num_outputs, num_filters, activation="relu", num_training_iterations=1000, checkpoint_frequency=5, master_seed=None, third_fc_layer=False):
        """
        Start the Training Process.

        :param num_outputs: (int) output space dimension/size
        :param num_training_iterations: (int) number of epochs to be trained for
        """

        view = self.loader_training_set.dataset.dataset.multi_view
        num_views = str(view).count("+") + 1 # as we cound additional views i.e., "yz" is 1 view, "+yz" is 2 views

        if view is None:
            print("Using the default Front View - xz Plane - for the Images.")
        else:
            print("Using the " + str(view) + " View Setting for the Images for Model Learning.")

        self.model = RegressorCNN(num_outputs,
                                  num_filters=num_filters,
                                  activation=activation,
                                  master_seed=master_seed,
                                  path_checkpoint_file=self.path_checkpoint_file,
                                  num_views=num_views,
                                  sequence_samples=self.loader_training_set.dataset.dataset.sequence_samples,
                                  third_fc_layer=third_fc_layer)

        if isinstance(self.prediction_target, list):
            prediction_target_name = "_".join(self.prediction_target)
        else:
            prediction_target_name = self.prediction_target

        self.model.train_model(num_training_iterations = num_training_iterations,
                               training_set_loader = self.loader_training_set,
                               prediction_target_name = prediction_target_name,
                               checkpoint_frequency = checkpoint_frequency,
                               validation_set_loader = self.loader_validation_set)
