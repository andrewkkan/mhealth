# Copyright 2022 FATHOM Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# References (https://archive.ics.uci.edu/ml/datasets/MHEALTH+Dataset):
#
# Banos, O., Garcia, R., Holgado, J. A., Damas, M., Pomares, H., Rojas, I., 
# Saez, A., Villalonga, C. mHealthDroid: a novel framework for agile development 
# of mobile health applications. Proceedings of the 6th International Work-conference 
# on Ambient Assisted Living an Active Ageing (IWAAL 2014), Belfast, Northern 
# Ireland, December 2-5, (2014).
#
# Banos, O., Villalonga, C., Garcia, R., Saez, A., Damas, M., Holgado, J. A., Lee, 
# S., Pomares, H., Rojas, I. Design, implementation and validation of a novel open 
# framework for agile development of mobile health applications. BioMedical Engineering 
# OnLine, vol. 14, no. S2:S6, pp. 1-20 (2015).
#
import sys
sys.path.append('./')

from absl import app, flags, logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union, Sequence

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

import os

from IPython import embed

FLAGS = flags.FLAGS

flags.DEFINE_string('logfile', '', 'Path and name of logfile.')
flags.DEFINE_string('csv_file', './data/mhealth_raw_data.csv', 'Path and name of csv file.')
flags.DEFINE_integer('federated_mode', 1, '1: Non-IID 10 subjects (no change from original dataset).  2: IID 10 subjects.  3: Centralized.')
flags.DEFINE_float('sample_duration_in_msec', 2500.0, 'Duration of each data sample in msecs.')
flags.DEFINE_float('test_data_size', 0.3, 'Portion of data allocated for test / validation.')
flags.DEFINE_integer('median_window_size', 10, 'Smoothening of raw data.  1 for no filtering.')

def median_filter(df: pd.DataFrame, target: pd.DataFrame, window: int):
    median = df.rolling(window).median().reset_index(drop=True).drop(np.arange(window-1))
    target = target.reset_index(drop=True).drop(np.arange(window-1))
    return median, target

def accuracy(confusion_matrix: np.ndarray):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements

class FederatedDataset(ABC):
    @abstractmethod
    def get_dataset_by_client(self, client:str) -> Tuple[Dataset]:
        # Outputs tuple of training data, test data
        pass
    def get_ndarray_by_client(self, client:str) -> Tuple[np.ndarray]:
        # Outputs tuple of training data, test data
        pass
    def get_client_list(self) -> List[str]:
        pass

class FederatedMHEALTHDataset(FederatedDataset):
    class MHEALTHDataset(Dataset):
        def __init__(self, dataset: Tuple[np.ndarray, np.ndarray], transform: Callable=None):
            self.data, self.labels = dataset
            self.len = len(self.data)
            self.randomize_order = np.random.randint(low=self.len, size=self.len)
            self.transform = transform

        def __len__(self):
            return self.len

        def __getitem__(self, idx):
            ridx = self.randomize_order[idx]
            if self.transform is not None:
                data: np.ndarray = self.transform(self.data[ridx])
            else:
                data: np.ndarray = self.data[ridx]
            label: int = int(self.labels[ridx])
            return data, label

    def __init__(self, 
        csv_file = './data/mhealth_raw_data.csv', 
        sample_duration_in_msec = 2500.0, # Duration of each data sample in msecs.
        federated_mode = 1, # 1: Non-IID 10 subjects (no change from original dataset).  2: IID 10 subjects.  3: Centralized.
        test_data_size = 0.3, # Portion of data allocated for test / validation.
        median_window_size = 5, # Smoothening of raw data.  1 for no filtering.
    ):
        # We use subject and client interchangeably.  
        # When it is related to the mhealth dataset, we try to use subject.
        # When it is related to federated learning, we try to use client.
        # First, create Non-IID datasets.
        if not os.path.exists(csv_file):
            import kaggle
            kaggle.api.authenticate()
            csv_path, _ = os.path.split(csv_file)
            kaggle.api.dataset_download_files('gaurav2022/mobile-health', path='csv_path', unzip=True)
        df = pd.read_csv(csv_file)
        row_freq = 50. # in Hz.  See https://archive.ics.uci.edu/ml/datasets/MHEALTH+Dataset
        sample_duration = int(np.ceil(sample_duration_in_msec * row_freq / float(1000))) # in rows
        unique_subjects = df.subject.unique()
        unique_activities = df.Activity.unique()
        self.tuple_per_subject = {}
        for subject in unique_subjects:
            df_subject = df.loc[df.subject == subject]
            df_target = df_subject.Activity
            df_subject = df_subject.drop(columns=["Activity","subject"])
            num_columns = df_subject.columns.size
            df_subject, df_target = median_filter(df_subject, df_target, median_window_size)
            df_subject = (df_subject - df_subject.mean()) / df_subject.std()
            df_subject["Activity"] = df_target
            dict_per_activity = {}
            min_activity_duration = np.inf
            min_duration_activity = None
            for activity in unique_activities:
                df_subject_activity = df_subject.loc[df_subject.Activity == activity]
                df_subject_activity = df_subject_activity.drop(columns=["Activity"])
                dict_per_activity[activity]: np.ndarray = df_subject_activity.values
                activity_duration = len(df_subject_activity)
                if activity_duration < min_activity_duration:
                    min_activity_duration = activity_duration
                    min_duration_activity = activity
            combined_sample_size = min_activity_duration - sample_duration + 1 # combined = train + test
            train_sample_size = int(combined_sample_size * (1. - test_data_size) + 0.5)
            test_sample_size = combined_sample_size - train_sample_size
            train_data_subject, train_labels_subject = (
                np.zeros((len(unique_activities)*train_sample_size, sample_duration, num_columns)),
                np.zeros((len(unique_activities)*train_sample_size)),
            )
            test_data_subject, test_labels_subject = (
                np.zeros((len(unique_activities)*test_sample_size, sample_duration, num_columns)),
                np.zeros((len(unique_activities)*test_sample_size)),
            )
            train_sample_counter, test_sample_counter = 0, 0
            for activity, data_activity in dict_per_activity.items():
                activity_duration = data_activity.shape[0]
                data_interval: float = (activity_duration - sample_duration) / float(combined_sample_size - 1) 
                test_samples_idx = np.random.choice(combined_sample_size, test_sample_size, replace = False)
                for sample_idx in range(combined_sample_size):
                    start_sample = int(sample_idx * data_interval + 0.5)
                    data_idx = np.arange(start_sample, start_sample+sample_duration)
                    if sample_idx in test_samples_idx:
                        test_data_subject[test_sample_counter, :, :] = data_activity[data_idx, :]
                        test_labels_subject[test_sample_counter] = activity
                        test_sample_counter += 1
                    else:
                        train_data_subject[train_sample_counter, :, :] = data_activity[data_idx, :]
                        train_labels_subject[train_sample_counter] = activity
                        train_sample_counter += 1
            self.tuple_per_subject[str(subject)] = ((train_data_subject, train_labels_subject), (test_data_subject, test_labels_subject))
        # For IID data, with same number of clients.  
        # We shuffle data samples from all clients then re-assign samples randomly back to clients.
        if federated_mode == 2:
            # 1. Pool data and labels from all subjects into activity bins
            # 2. Randomize indices for each activity
            # 3. For each subject, use randomized indices to select data from each activity bin
            stacked_train_data, stacked_train_labels = None, None
            stacked_test_data, stacked_test_labels = None, None
            for subject in unique_subjects:
                if stacked_train_labels is None:
                    stacked_train_data, stacked_train_labels = self.tuple_per_subject[str(subject)][0]
                    stacked_test_data, stacked_test_labels =  self.tuple_per_subject[str(subject)][1]
                else:
                    stacked_train_data = np.append(stacked_train_data, self.tuple_per_subject[str(subject)][0][0], axis = 0)
                    stacked_train_labels = np.append(stacked_train_labels, self.tuple_per_subject[str(subject)][0][1], axis = 0)
                    stacked_test_data = np.append(stacked_test_data, self.tuple_per_subject[str(subject)][1][0], axis = 0)
                    stacked_test_labels = np.append(stacked_test_labels, self.tuple_per_subject[str(subject)][1][1], axis = 0)
            rndidx_per_activity = {}
            for activity in unique_activities:
                train_idx, test_idx = np.argwhere(stacked_train_labels == activity).flatten(), np.argwhere(stacked_test_labels == activity).flatten()
                np.random.shuffle(train_idx)
                np.random.shuffle(test_idx)
                rndidx_per_activity[activity] = (train_idx, test_idx)
            for subject in unique_subjects:
                train_data_idx, test_data_idx = 0, 0
                for activity in unique_activities:
                    num_train_samples = np.argwhere(self.tuple_per_subject[str(subject)][0][1] == activity).flatten().size
                    num_test_samples = np.argwhere(self.tuple_per_subject[str(subject)][1][1] == activity).flatten().size
                    self.tuple_per_subject[str(subject)][0][0][train_data_idx:train_data_idx+num_train_samples, :] = \
                        stacked_train_data[rndidx_per_activity[activity][0][0:num_train_samples], :]
                    self.tuple_per_subject[str(subject)][1][0][test_data_idx:test_data_idx+num_test_samples, :] = \
                        stacked_test_data[rndidx_per_activity[activity][1][0:num_test_samples], :]
                    self.tuple_per_subject[str(subject)][0][1][train_data_idx:train_data_idx+num_train_samples] = activity
                    self.tuple_per_subject[str(subject)][1][1][test_data_idx:test_data_idx+num_test_samples] = activity
                    rndidx_per_activity[activity] = (
                        rndidx_per_activity[activity][0][num_train_samples:],
                        rndidx_per_activity[activity][1][num_test_samples:],
                    )
                    train_data_idx += num_train_samples
                    test_data_idx += num_test_samples
        # For centralized data with 1 single client "central_server".
        elif federated_mode == 3:
            # All we need is pool data and labels from all subjects and activities into a single bin.
            stacked_train_data, stacked_train_labels = None, None
            stacked_test_data, stacked_test_labels = None, None
            for subject in unique_subjects:
                if stacked_train_labels is None:
                    stacked_train_data, stacked_train_labels = self.tuple_per_subject[str(subject)][0]
                    stacked_test_data, stacked_test_labels =  self.tuple_per_subject[str(subject)][1]
                else:
                    stacked_train_data = np.append(stacked_train_data, self.tuple_per_subject[str(subject)][0][0], axis = 0)
                    stacked_train_labels = np.append(stacked_train_labels, self.tuple_per_subject[str(subject)][0][1], axis = 0)
                    stacked_test_data = np.append(stacked_test_data, self.tuple_per_subject[str(subject)][1][0], axis = 0)
                    stacked_test_labels = np.append(stacked_test_labels, self.tuple_per_subject[str(subject)][1][1], axis = 0)
            # Let's rename our only subject to "central_server" for centralized data
            self.tuple_per_subject = {'central_server': (
                (stacked_train_data, stacked_train_labels),
                (stacked_test_data, stacked_test_labels),
            )}

    def get_dataset_by_client(self, client: str) -> Tuple[MHEALTHDataset]:
        return (
            self.MHEALTHDataset(dataset = self.tuple_per_subject[client][0]),
            self.MHEALTHDataset(dataset = self.tuple_per_subject[client][1]),
        )

    def get_ndarray_by_client(self, client: str) -> Tuple[np.ndarray]:
        return self.tuple_per_subject[client]

    def get_client_list(self) -> List[str]:
        return list(self.dataset_per_subject.keys())

def main(argv: Sequence[str]) -> None:

    if FLAGS.logfile:
        logdir, logfn = os.path.split(FLAGS.logfile)
        logging.use_absl_handler()
        logging.get_absl_handler().use_absl_log_file(logfn, logdir) 
        logging.info(FLAGS.flag_values_dict())

    del argv

    mhealth_dataset = FederatedMHEALTHDataset(
        csv_file = FLAGS.csv_file,
        sample_duration_in_msec = FLAGS.sample_duration_in_msec,
        federated_mode = FLAGS.federated_mode,
        test_data_size = FLAGS.test_data_size,
        median_window_size = FLAGS.median_window_size,
    )

    dataset = mhealth_dataset.get_ndarray_by_client('central_server')
    x_train = dataset[0][0]
    x_test = dataset[1][0]
    y_train = dataset[0][1]
    y_test = dataset[1][1]

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    from sklearn.linear_model import LogisticRegression
    # C =1000
    # logistic_regression= LogisticRegression(max_iter=50000, C=C)
    # logistic_regression.fit(x_train,y_train)
    # y_pred=logistic_regression.predict(x_test)
    # diff_vec = y_pred != y_test
    # err = int(1000*sum(diff_vec)/len(y_pred))/10
    # print("LogisticRegression: C", C, "error ", err, "%")   

    from sklearn.neural_network import MLPClassifier
    classifier = MLPClassifier(
        hidden_layer_sizes=(75,100,25), 
        max_iter=1000,
        activation = 'relu',
        solver='adam',
        random_state=1
    )
    classifier.fit(x_train,y_train)
    y_pred=classifier.predict(x_test)
    diff_vec = y_pred != y_test
    err = int(1000*sum(diff_vec)/len(y_pred))/10
    print("MLPClassifier: error ", err, "%")   

    #Importing Confusion Matrix
    from sklearn.metrics import confusion_matrix
    #Comparing the predictions against the actual observations in y_val
    cm: np.ndarray = confusion_matrix(y_pred, y_test)
    #Printing the accuracy
    print("Accuracy of MLPClassifier : ", accuracy(cm))

if __name__ == '__main__':
    app.run(main)
