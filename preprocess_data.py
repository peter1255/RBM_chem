import pandas as pd
import numpy as np
import gc
import random

ROOT_DIR=''
TRAIN_SAMPLES = 8000
TEST_SAMPLES = 2000
PARTITIONS = 30
PARAMS = 3


class Partition:

    def __init__(self, train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file

    @staticmethod
    def partition(num_part, lower, upper):
        """
        Partitions a range into num_part partitions of uniform size.
        Lower specifies lower bound, upper specifies upper bound
        """
        partition_size = (upper - lower) / num_part
        partition_ = []
        for x in range(num_part):
            partition_.append(min((x * partition_size) + lower, upper))
        return partition_

    def get_partitions(self, data_set):
        """
        Gets the partitions for the data based on the training set.
        It simply divides the range into PARTITIONS number of partitions.
        """
        parts = np.zeros([PARAMS, PARTITIONS], dtype=float)
        for i in range(len(data_set[0])-1): # minus one due to index
            column = data_set[:,i+1]
            min_ = np.amin(column)
            max_ = np.amax(column)
            parts[i,:] = self.partition(PARTITIONS, min_, max_)
        return parts

    @staticmethod
    def place_in_partition(value, part):
        """
        Places value into a partition into part, returns vector of all zeroes except for value with one.
        If test argument is set to True then it will randomly remove 1/3 of the data and replace with -1
        """
        pert = part.copy()
        written = False
        if value == -1:
            pert = np.full([PARTITIONS], -1)
        else:
            for i in range(len(part)-1,-1,-1):
                if value >= pert[i] and written == False:
                    pert[i] = 1
                    written = True
                else:
                    pert[i] = 0
        return pert

    def partition_dataset(self, array, SAMPLES):
        """
        Converts every entry of array from a double to a partition.
        Assumes that first column is index.
        """

        part_table = np.zeros([SAMPLES, PARAMS, PARTITIONS], dtype=int)

        for j in range(PARAMS):
            column = array[:,j+1]
            for i in range(SAMPLES):
                part_table[i, j] = self.place_in_partition(column[i], self.parts[j])

        key = part_table.reshape((part_table.shape[0], -1)).copy()

        for row in part_table:
            row[random.choice(range(PARAMS))] = np.full(PARTITIONS, -1)
        part_table = part_table.reshape((part_table.shape[0], -1))

        return key, part_table


    def get_dataset(self):

        gc.enable()

        # opening data files and converting to numpy arrays
        training_set = pd.read_csv(self.train_file, sep='\t', header=None, engine='python', encoding='latin-1')
        training_set = np.array(training_set, dtype=float)

        test_set = pd.read_csv(self.test_file, sep='\t', header=None, engine='python', encoding='latin-1')
        test_set = np.array(test_set, dtype=float)

        self.parts = self.get_partitions(training_set)
        np.savetxt('part.txt', self.parts)

        _,training_set = self.partition_dataset(training_set, TRAIN_SAMPLES)
        reference_test_set, test_set_ = self.partition_dataset(test_set, TEST_SAMPLES)
        #= partition_dataset(test_set, TEST_SAMPLES)

        return training_set, test_set_, reference_test_set
