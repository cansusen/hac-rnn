from abc import ABCMeta, abstractmethod
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import os
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.platform import gfile
import tarfile
import gzip
import numpy as np
import tensorflow as tf
import pandas
import pickle
import sys
import csv

#-------------------------------------------------------------------------
'''
    This file includes a collection of dataset classes. 
    A `dataset` class should inherit the abstract class `dataset`. 
'''

#-------------------------------------------------------------------------
class dataset():
    '''
        dataset is a abstract class defining the APIs (e.g., load functions) for dataset classes
    '''
    __metaclass__ = ABCMeta

    def __init__(self, 
                 feature_shape, # the shape of an input instance
                 n_class = 1, # number of classes in the output
                 feature_dtype = tf.float32 # the data type of the feature vector
                 ): 
        self._feature_shape = feature_shape
        self._n_class = n_class
        self._feature_dtype = feature_dtype

        if not tf.gfile.Exists(self.path):
            tf.gfile.MakeDirs(self.path)   

    # ----------------------------------------------
    @abstractmethod
    def train_batch(self,batch_size):
        '''
            get a tensor of training batch
            Input: 
                batch_size: the size of the batch
            Return:
                a tensor of shape: [batch_size, ... feature_shape ...]
        '''
        raise NotImplementedError
    
    def test_batch(self,batch_size):
        '''
            get a tensor of test batch
            Input: 
                batch_size: the size of the batch
            Return:
                a tensor of shape: [batch_size, ... feature_shape ...]
        '''
        raise NotImplementedError
    
    # ----------------------------------------------
    @classmethod
    def class_name(cls):
        return cls.__name__

    @property
    def name(self):
        return self.class_name()

    @property
    def path(self):
        '''the path of the dataset files to be saved'''
        return os.path.join('datasets',self.name)
    
    @property
    def n_class(self):
        return self._n_class
    
    # ----------------------------------------------
    '''utility functions for TFRecords'''
    # ----------------------------------------------
    @staticmethod
    def _int64_feature(value):
        """
        Returns a TF-Feature of int64.
        Args:
            values: A string.
        Returns:
            a TF-Feature.
        """
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    # ----------------------------------------------
    @staticmethod
    def _bytes_feature(value):
        """
        Returns a TF-Feature of bytes.
        Args:
            values: A string.
        Returns:
            a TF-Feature.
        """
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    # ----------------------------------------------
    @staticmethod
    def dump_tfrecord(x, y, filepath):
        """
        save dataset as binary TFRecord file 
        Args:
            x: a list/iterator of feature vectors
            y: a list/iterator of labels (if no label, use None)
            filepath: the path and file name for the TFRecord file to save 
        """
        with tf.python_io.TFRecordWriter(filepath) as writer:
            for xi, yi in map(None, x, y):
                x_str = xi.tostring()
                if yi is None:
                    yi = 0
                features= tf.train.Features(
                        feature={'y': dataset._int64_feature(int(yi)),
                                 'x': dataset._bytes_feature(x_str)})
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())
    
    # ----------------------------------------------
    @staticmethod
    def load_tfrecord(filepath,
                      feature_dtype,
                      feature_shape):
        ''' 
        Read TFRecord file to get one data sample tensor.
        Args:
            filepath: the path and file name of the TFRecord file.
            feature_dtype: the data type of the features (e.g., float32, int32) 
            feature_shape: the shape of the features (e.g., [28,28,1] in MNIST dataset) 
        Returns:
            x: a tensor for the features of one data record (sample)
            label: a tensor for the label of one data record
        Reference: https://www.tensorflow.org/programmers_guide/reading_data
        '''
        filename_queue = tf.train.string_input_producer([filepath])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue) 
        features = tf.parse_single_example(serialized_example, 
                                       features={
                                           'y': tf.FixedLenFeature([], tf.int64),
                                           'x' : tf.FixedLenFeature([], tf.string),
                                       })
        x = tf.decode_raw(features['x'], feature_dtype)
        x = tf.reshape(x, feature_shape)
        
        label = tf.cast(features['y'], tf.int64)
        return x, label
    
    # ----------------------------------------------
    @staticmethod
    def load_batch( batch_size,
                    filepath, 
                    feature_dtype,
                    feature_shape,
                    capacity =2000,
                    min_queue_examples = 50,
                    n_threads=20):
        """
        Construct a queued batch of data and labels.
        Args:
            batch_size: Number of images per batch.
            filepath: the path and file name of the TFRecord file 
            feature_dtype: the data type of the features (e.g., float32, int32) 
            feature_shape: the shape of the features (e.g., [28,28,1] in MNIST dataset) 
            capacity: the capacity of the queue
            min_queue_examples: int32, minimum number of samples to retain
              in the queue that provides of batches of examples.
            n_threads: the number of threads to feed data into the queue 
        Returns:
            x_batch: Feature tensor of shape [batch_size, ...feature_shape...].
            label_batch: Label tensor of shape [batch_size].
        Reference: https://www.tensorflow.org/programmers_guide/reading_data
        """
        # load a queued example tensor
        x, label = dataset.load_tfrecord(   filepath,
                                            feature_dtype,
                                            feature_shape)


        # load a tensor of a batch of randomly shuffle examples
        shuffle = False
        if shuffle: 
            x_batch, label_batch = tf.train.shuffle_batch([x, label],
                                                batch_size=batch_size,
                                                capacity=capacity,
                                                min_after_dequeue=min_queue_examples,
                                                num_threads=n_threads)
            print('batch shuffled!!')
        else:
            x_batch, label_batch = tf.train.batch([x, label],
                                                batch_size=batch_size,
                                                capacity=capacity,
                                                num_threads=n_threads)
            print('batch not shuffled!!')
            print(x_batch)
            print(label_batch)

        return x_batch, label_batch
    # ----------------------------------------------
    def train_batch(self, batch_size,n_threads=19):
        """
        Construct a queued batch from training set.
        Args:
            batch_size: Number of images per batch.
            n_threads: the number of threads to feed data into the queue 
        Returns:
            x_batch: Feature tensor of shape [batch_size, ...feature_shape...].
            label_batch: Label tensor of shape [batch_size].
        Reference: https://www.tensorflow.org/programmers_guide/reading_data
        """
        filename = os.path.join(self.path, '4.train.tfrecords')
        with tf.name_scope('train'):
            return self.load_batch(batch_size,
                                   filename,
                                   self._feature_dtype,
                                   self._feature_shape,
                                   n_threads=n_threads)
        
    # ----------------------------------------------
    def test_batch(self, batch_size,n_threads=1):
        """
        Construct a queued batch from test set.
        Args:
            batch_size: Number of images per batch.
            n_threads: the number of threads to feed data into the queue 
        Returns:
            x_batch: Feature tensor of shape [batch_size, ...feature_shape...].
            label_batch: Label tensor of shape [batch_size].
        Reference: https://www.tensorflow.org/programmers_guide/reading_data
        """
        filename = os.path.join(self.path, '4.test.tfrecords')
        with tf.name_scope('test'):
            return self.load_batch(batch_size, #687,#2190,#646,#batch_size,
                                   filename,
                                   self._feature_dtype,
                                   self._feature_shape,
                                   n_threads=n_threads)

            
#-------------------------------------------------------------------------            

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# Clinical Time Series       
class ClinicalTS_combined_1(dataset):
    
    def __init__(self):
        super(ClinicalTS_combined_1, self).__init__(feature_shape = [1,5000], n_class = 2)
        trainfile = os.path.join(self.path, 'train.tfrecords')
        testfile = os.path.join(self.path, 'test.tfrecords')
    
        datasets = self.load_datasets()

        # save to TFRecord
        self.dump_tfrecord(datasets[0][0],datasets[0][1],trainfile) 
        self.dump_tfrecord(datasets[1][0],datasets[1][1],testfile)
        
    def load_datasets(self):
        #data_dir = os.path.join(self.path, 'cdif_2000_bow')
        data_dir = self.path 
        
        X_train_path = os.path.join(data_dir, 'BOW_combined_X_train.npy')
        X_test_path  = os.path.join(data_dir, 'BOW_combined_X_test.npy')
        y_train_path = os.path.join(data_dir, 'BOW_combined_y_train.npy')
        y_test_path  = os.path.join(data_dir, 'BOW_combined_y_test.npy')
            
        X_train = np.load(X_train_path)
        X_test  = np.load(X_test_path)
        X_train = X_train.astype(np.float32)
        X_test  = X_test.astype(np.float32)
       
        y_train = np.load(y_train_path)
        y_test  = np.load(y_test_path)
        
        return [(X_train, y_train), (X_test, y_test)]
#-------------------------------------------------------------------------
class ClinicalTS_combined(dataset):
    
    def __init__(self, disease = 'cdif-rnn'):
        self.disease = 'cdif-rnn'
        super(ClinicalTS_combined, self).__init__(feature_shape = [6000], n_class = 2, feature_dtype = tf.int32)
        
        
        trainfile = os.path.join(self.path, '4.train.tfrecords')
        testfile = os.path.join(self.path, '4.test.tfrecords')
    
        datasets = self.load_datasets()
        
        #filename = os.path.join(self.path, 'vocab.pkl')
        #with open(filename, 'r') as f:
        #    self.vocab = pickle.load(f)
        
        self.max_seq_len = 6000
        self.char_based = False

        # save to TFRecord
        self.dump_tfrecord(datasets[0][0],datasets[0][1],trainfile) 
        self.dump_tfrecord(datasets[1][0],datasets[1][1],testfile)
        
    def load_datasets(self):
        data_dir = self.path 
        
        X_train_path = os.path.join(data_dir, 'cdif/Words_combined_X_train.npy')
        X_test_path  = os.path.join(data_dir, 'cdif/Words_combined_X_test.npy')
        y_train_path = os.path.join(data_dir, 'cdif/Words_combined_y_train.npy')
        y_test_path  = os.path.join(data_dir, 'cdif/Words_combined_y_test.npy')
            
        X_train = np.load(X_train_path)
        X_test  = np.load(X_test_path)
        X_train = X_train.astype(np.int32)
        X_test  = X_test.astype(np.int32)
        
        y_train = np.load(y_train_path)
        y_test  = np.load(y_test_path)
        
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)
        print(X_train[0])
        return [(X_train, y_train), (X_test, y_test)]

   
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# Clinical Time Series       
class ClinicalTS_individual_1(dataset):
    
    def __init__(self,disease = 'mort-h2a-complex'):
        self.disease = 'mort-h2a-complex'
        super(ClinicalTS_individual_1, self).__init__(feature_shape = [20, 300], n_class = 2, feature_dtype = tf.int32)
        
        #init_path = 'datasets/ClinicalTS/ClinicalTS_individual/'
        
        trainfile = os.path.join(self.path, '4.train.tfrecords')
        testfile = os.path.join(self.path, '4.test.tfrecords')
    
        datasets = self.load_datasets()
        
        #filename = os.path.join(self.path, 'vocab.pkl')
        #with open(filename, 'r') as f:
        #    self.vocab = pickle.load(f)
        
        self.max_seq_len = 300
        self.char_based = False

        # save to TFRecord
        self.dump_tfrecord(datasets[0][0],datasets[0][1],trainfile) 
        self.dump_tfrecord(datasets[1][0],datasets[1][1],testfile)
        
    def load_datasets(self):
        data_dir = self.path 
        
        X_train_path = os.path.join(data_dir, 'mort/Words_individual_X_train.npy')
        X_test_path  = os.path.join(data_dir, 'mort/Words_individual_X_test.npy')
        y_train_path = os.path.join(data_dir, 'mort/Words_individual_y_train.npy')
        y_test_path  = os.path.join(data_dir, 'mort/Words_individual_y_test.npy')
            
        X_train = np.load(X_train_path)
        X_test  = np.load(X_test_path)
        X_train = X_train.astype(np.int32)
        X_test  = X_test.astype(np.int32)
        
        y_train = np.load(y_train_path)
        y_test  = np.load(y_test_path)
        
        return [(X_train, y_train), (X_test, y_test)]

#-------------------------------------------------------------------------
# Clinical Notes - Hierarchical Attention - Enhanced with Time variables       
class ClinicalTS_hier_time(dataset):
    
    def __init__(self,disease = 'cdif-all-features'):
        self.disease = 'cdif-all-features'
        super(ClinicalTS_hier_time, self).__init__(feature_shape = [21, 321], n_class = 2, feature_dtype = tf.float32)
        
        trainfile = os.path.join(self.path, '4.train.tfrecords')
        testfile = os.path.join(self.path, '4.test.tfrecords')
    
        datasets = self.load_datasets()
        
        # save to TFRecord
        self.dump_tfrecord(datasets[0][0],datasets[0][1],trainfile) 
        self.dump_tfrecord(datasets[1][0],datasets[1][1],testfile)
        
    def load_datasets(self):
        data_dir = self.path 
        
        X_train_path = os.path.join(data_dir, 'cdif-age/Words_individual_X_train.npy')
        X_test_path  = os.path.join(data_dir, 'cdif-age/Words_individual_X_test.npy')
        y_train_path = os.path.join(data_dir, 'cdif-age/Words_individual_y_train.npy')
        y_test_path  = os.path.join(data_dir, 'cdif-age/Words_individual_y_test.npy')
            
        X_train = np.load(X_train_path)
        X_test  = np.load(X_test_path)
        y_train = np.load(y_train_path)
        y_test  = np.load(y_test_path)
        
        X_train = X_train.astype(np.float32)
        X_test  = X_test.astype(np.float32)

        return [(X_train, y_train), (X_test, y_test)]

#
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# Clinical Time Series       
class ClinicalTS_numeric_combined(dataset):
    
    def __init__(self, disease = 'cdif'):
        self.disease = 'cdif'
        super(ClinicalTS_numeric_combined, self).__init__(feature_shape = [32], n_class = 2)
                
        trainfile = os.path.join(self.path, '4.train.tfrecords')
        testfile = os.path.join(self.path, '4.test.tfrecords')
    
        datasets = self.load_datasets()

        # save to TFRecord
        self.dump_tfrecord(datasets[0][0],datasets[0][1],trainfile) 
        self.dump_tfrecord(datasets[1][0],datasets[1][1],testfile)
        
    def load_datasets(self):
        data_dir = self.path 
        
        X_train_path = os.path.join(data_dir, 'mort/Avg_combined_X_train.npy')
        X_test_path  = os.path.join(data_dir, 'mort/Avg_combined_X_test.npy')
        y_train_path = os.path.join(data_dir, 'mort/Avg_combined_y_train.npy')
        y_test_path  = os.path.join(data_dir, 'mort/Avg_combined_y_test.npy')
            
        X_train = np.load(X_train_path)
        X_test  = np.load(X_test_path)
        X_train = X_train.astype(np.float32)
        X_test  = X_test.astype(np.float32)
        
        y_train = np.load(y_train_path)
        y_test  = np.load(y_test_path)
        
        return [(X_train, y_train), (X_test, y_test)]

#-------------------------------------------------------------------------
# Clinical Time Series       
class ClinicalTS_numeric_sequential(dataset):
    
    def __init__(self, disease = 'mort-run1'):
        self.disease = 'mort-run1'
        super(ClinicalTS_numeric_sequential, self).__init__(feature_shape = [20,32], n_class = 2)

        trainfile = os.path.join(self.path, '4.train.tfrecords')
        testfile = os.path.join(self.path, '4.test.tfrecords')
    
        datasets = self.load_datasets()

        # save to TFRecord
        self.dump_tfrecord(datasets[0][0],datasets[0][1],trainfile) 
        self.dump_tfrecord(datasets[1][0],datasets[1][1],testfile)
        
    def load_datasets(self):
        data_dir = self.path 
        
        X_train_path = os.path.join(data_dir, 'mort/Avg_individual_X_train.npy')
        X_test_path  = os.path.join(data_dir, 'mort/Avg_individual_X_test.npy')
        y_train_path = os.path.join(data_dir, 'mort/Avg_individual_y_train.npy')
        y_test_path  = os.path.join(data_dir, 'mort/Avg_individual_y_test.npy')
            
        X_train = np.load(X_train_path)
        X_test  = np.load(X_test_path)
        X_train = X_train.astype(np.float32)
        X_test  = X_test.astype(np.float32)
        
        y_train = np.load(y_train_path)
        y_test  = np.load(y_test_path)
        
        return [(X_train, y_train), (X_test, y_test)]

#-------------------------------------------------------------------------
# Yelp dataset experiments
#-------------------------------------------------------------------------   
class yelp(dataset):
    
    def __init__(self, disease = 'yelp'):
        self.disease = 'yelp'
        super(yelp, self).__init__(feature_shape = [50,100], n_class = 2)

        trainfile = os.path.join(self.path, '4.train.tfrecords')
        testfile = os.path.join(self.path, '4.test.tfrecords')
    
        datasets = self.load_datasets()

        # save to TFRecord
        self.dump_tfrecord(datasets[0][0],datasets[0][1],trainfile) 
        self.dump_tfrecord(datasets[1][0],datasets[1][1],testfile)
        
    def load_datasets(self):
        data_dir = self.path 
        
        #X_train_path = os.path.join(data_dir, 'output_reviews_train.npy')
        #X_test_path  = os.path.join(data_dir, 'output_reviews_dev.npy')
        #y_train_path = os.path.join(data_dir, 'labels_train.npy')
        #y_test_path  = os.path.join(data_dir, 'labels_dev.npy')
        X_train_path = os.path.join(data_dir, 'output_reviews_train.npy')
        X_test_path  = os.path.join(data_dir, 'output_reviews_50.npy')
        y_train_path = os.path.join(data_dir, 'labels_train.npy')
        y_test_path  = os.path.join(data_dir, 'labels_50.npy')
            
        X_train = np.load(X_train_path)
        X_test  = np.load(X_test_path)
        X_train = X_train.astype(np.float32)
        X_test  = X_test.astype(np.float32)
        
        y_train = np.load(y_train_path)
        y_test  = np.load(y_test_path)
        
        return [(X_train, y_train), (X_test, y_test)]


#-------------------------------------------------------------------------
# TIME EXPERIMENTS - DUPLICATE CLASSES OF ClinicalTS_numeric_sequential
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# Clinical Time Series       
class ClinicalTS_bow_sequential(dataset):
    
    def __init__(self,disease = 'mort'):
        self.disease = 'mort'
        #super(ClinicalTS_bow_sequential, self).__init__(feature_shape = [10,7000], n_class = 2)
        super(ClinicalTS_bow_sequential, self).__init__(feature_shape = [20,300], n_class = 2)

        trainfile = os.path.join(self.path, '4.train.tfrecords')
        testfile = os.path.join(self.path, '4.test.tfrecords')
    
        datasets = self.load_datasets()

        # save to TFRecord
        self.dump_tfrecord(datasets[0][0],datasets[0][1],trainfile) 
        self.dump_tfrecord(datasets[1][0],datasets[1][1],testfile)
        
    def load_datasets(self):
        data_dir = self.path 
        
        X_train_path = os.path.join(data_dir, 'mort/BOW_individual_X_train.npy')
        X_test_path  = os.path.join(data_dir, 'mort/BOW_individual_X_test.npy')
        y_train_path = os.path.join(data_dir, 'mort/BOW_individual_y_train.npy')
        y_test_path  = os.path.join(data_dir, 'mort/BOW_individual_y_test.npy')
            
        X_train = np.load(X_train_path)
        X_test  = np.load(X_test_path)
        X_train = X_train.astype(np.float32)
        X_test  = X_test.astype(np.float32)
        
        y_train = np.load(y_train_path)
        y_test  = np.load(y_test_path)
        
        return [(X_train, y_train), (X_test, y_test)]

#-------------------------------------------------------------------------
def generate_toy_regular():
    # This toy dataset is designed to emphasize between-document relationships 
    t1 = [[0, 0], [0, 0], [0, 0]]
    y1 = 1
    t2 = [[0, 0], [1, 0], [0, 0]]
    y2 = 0

    notes = []
    labels = []

    for i in range(100):
        notes.append(t1)
        labels.append(y1)
        notes.append(t2)
        labels.append(y2)

    X_train = notes
    X_test  = X_train
    y_train = labels
    y_test  = labels
    print(X_train[0])
    print(X_test[0])
    return [(X_train, y_train), (X_test, y_test)]

def generate_toy_hierarchical():
    # This toy dataset is designed to emphasize between-document relationships 
    t1 = [[2, 2, 2, 0], [2, 2, 1, 2], [2, 2, 2, 0]]
    y1 = 1
    t2 = [[2, 2, 2, 2], [2, 1, 2, 0], [2, 2, 2, 0]]
    y2 = 0

    notes = []
    labels = []

    for i in range(4):
        notes.append(t1)
        labels.append(y1)
        notes.append(t2)
        labels.append(y2)

    notes  = np.array(notes)
    labels = np.array(labels)
    X_train = notes
    X_test  = X_train
    y_train = labels
    y_test  = labels

    return [(X_train, y_train), (X_test, y_test)]

def get_sequence_lenght_b(X):
    """
      Input X. Outputs the actual sequence lenghts for dynamic rnn
    """

    out = np.zeros([X.shape[0], X.shape[1]])
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            mylist = X[i,j]
            out[i,j]= len(mylist) - next((i for i, x in enumerate(reversed(mylist)) if x), len(mylist))
    print(out)
    return out

class ClinicalTS_toy(dataset):
    
    def __init__(self):
        super(ClinicalTS_toy, self).__init__(feature_shape = [3,4], n_class = 2) #ts, vector size for each timestamp  -Num docs, num words 1,9
        
        
        trainfile = os.path.join(self.path, 'train.tfrecords')
        testfile = os.path.join(self.path, 'test.tfrecords')
    
        #datasets = generate_toy_regular()
        datasets = generate_toy_hierarchical()
        self.seq_len  = get_sequence_lenght_b(datasets[0][0])

        # save to TFRecord
        self.dump_tfrecord(datasets[0][0],datasets[0][1],trainfile) 
        self.dump_tfrecord(datasets[1][0],datasets[1][1],testfile)
        

