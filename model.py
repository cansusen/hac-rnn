from abc import ABCMeta, abstractmethod
import numpy as np
import tensorflow as tf
import os
import math
import random
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import dynamic_rnn as dyn_rnn

#-------------------------------------------------------
class model(object):
    '''
        model is an abstract class for deep learning models.
        It defines the inference/loss pattern for model building.
        1. inference() - Builds the model as far as is required for running the network
        forward to make predictions.
        2. loss() - Adds to the inference model the layers required to generate loss.
    '''
    __metaclass__ = ABCMeta
    
    # ----------------------------------------------
    # default values for properties
    def __init__(self,
                 n_class = 10, # number of categories in classification task
                 batch_size = 16, # number of samples in each mini-batch
                 optimizer = 'adam', # 'sgd', 'adam', 'rmsprop'
                 learning_rate = 0.001, #0.01, # initial learning rate
                 learning_rate_decay_rate= 0.9, # exponential decay to the learning rate
                 learning_rate_decay_steps= 10000, # how many steps to decay learning rate once
                 rmsprop_decay = 0.9, #for rmsprop: Discounting factor for the history/coming gradient
                 adam_beta1=0.5, # for adam: 
                 dtype=tf.float32, # data type: tf.float32, tf.float16
                 l2_regularization = 1e-5, # weight for l2 regularization: 0.001 or None
                 cpu = True, # whether or not to place the variables on CPU
                 embedding_dim = 4, # the number of dimensions in embedding layer
                 nce = False, # whether or not need to add NCE (noise-contrastive estimation) to the embedding layer 
                 nce_nsample = 64, # the number of negative samples in NCE
                 nce_weight = 1e-2, # the weight for NCE loss  
                 grad_clip = False, # whether or not to use gradient clipping
                 grad_clip_range =[-100,100], # the range for gradient clipping
                 grad_clip_type ='value', # the type of gradient clipping, by valueor norm
                 char_based = False # whether or not to use char-based model
                ):
        self._n_class = n_class
        self._batch_size = batch_size
        self._optimizer = optimizer
        self._learning_rate =learning_rate
        self._learning_rate_decay_rate = learning_rate_decay_rate
        self._learning_rate_decay_steps = learning_rate_decay_steps
        self._rmsprop_decay = rmsprop_decay
        self._adam_beta1 = adam_beta1
        self._dtype = dtype
        self._l2_regularization = l2_regularization
        self._cpu = cpu
        self._embedding_dim = embedding_dim
        self._nce = nce 
        self._nce_nsample = nce_nsample 
        self._nce_weight = nce_weight
        self._char_based = char_based
        self._grad_clip = grad_clip
        self._grad_clip_range = grad_clip_range
        self._grad_clip_type = grad_clip_type


    # ----------------------------------------------
    # list of properties and their setters
    @classmethod
    def class_name(cls):
        return cls.__name__
    @property
    def name(self):
        return self.class_name()

    @property
    def n_class(self):
        return self._n_class
    @n_class.setter
    def n_class(self,value):
        self._n_class = value
        
    @property
    def batch_size(self):
        return self._batch_size
    @batch_size.setter
    def batch_size(self,value):
        self._batch_size = value
        
    @property
    def optimizer(self):
        return self._optimizer  
    @optimizer.setter
    def optimizer(self,value):
        self._optimizer = value
    
    @property
    def learning_rate(self):
        return self._learning_rate
    @learning_rate.setter
    def learning_rate(self,value):
        self._learning_rate = value
    
    @property
    def learning_rate_decay_rate(self):
        return self._learning_rate_decay_rate
    @learning_rate_decay_rate.setter
    def learning_rate_decay_rate(self,value):
        self._learning_rate_decay_rate = value
    
    @property
    def learning_rate_decay_steps(self):
        return self._learning_rate_decay_steps
    @learning_rate_decay_steps.setter
    def learning_rate_decay_steps(self,value):
        self._learning_rate_decay_steps = value
    
    @property
    def rmsprop_decay(self):
        return self._rmsprop_decay
    @rmsprop_decay.setter
    def rmsprop_decay(self,value):
        self._rmsprop_decay = value
    
    @property
    def adam_beta1(self):
        return self._adam_beta1
    @adam_beta1.setter
    def adam_beta1(self,value):
        self._adam_beta1 = value
    
    @property
    def l2_regularization(self):
        return self._l2_regularization
    @l2_regularization.setter
    def l2_regularization(self,value):
        self._l2_regularization = value
    
    @property
    def dtype(self):
        return self._dtype
    @dtype.setter
    def dtype(self,value):
        self._dtype=value
        
    @property
    def global_step(self):
        return self._global_step
    @global_step.setter
    def global_step(self,value):
        self._global_step=value
        
    @property
    def plot_gradient(self):
        return self._plot_gradient
    @plot_gradient.setter
    def plot_gradient(self,value):
        self._plot_gradient=value
        
    @property
    def embedding_dim(self):
        return self._embedding_dim
    @embedding_dim.setter
    def embedding_dim(self,value):
        self._embedding_dim=value    
        
    @property
    def nce(self):
        return self._nce
    @nce.setter
    def nce(self,value):
        self._nce=value    
 
    @property
    def nce_nsample(self):
        return self._nce_nsample
    @nce_nsample.setter
    def nce_nsample(self,value):
        self._nce_nsample=value    

    @property
    def nce_weight(self):
        return self._nce_weight
    @nce_weight.setter
    def nce_weight(self,value):
        self._nce_weight = value         

    @property
    def cpu(self):
        return self._cpu
    @cpu.setter
    def cpu(self,value):
        self._cpu=value       
        
    @property
    def vocab(self):
        return self._vocab
    @vocab.setter
    def vocab(self,value):
        self._vocab = value

    @property
    def char_based(self):
        return self._char_based
    @char_based.setter
    def char_based(self,value):
        self._char_based = value

    @property
    def grad_clip(self):
        return self._grad_clip
    @grad_clip.setter
    def grad_clip(self,value):
        self._grad_clip = value

    @property
    def grad_clip_range(self):
        return self._grad_clip_range
    @grad_clip_range.setter
    def grad_clip_range(self,value):
        self._grad_clip_range = value 

    @property
    def grad_clip_type(self):
        return self._grad_clip_type
    @grad_clip_type.setter
    def grad_clip_type(self,value):
        self._grad_clip_type = value 

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
    # ----------------------------------------------
    @abstractmethod
    def inference(self, x):
        """
              Abstract method: Build the model up to where it may be used for inference.
              Args:
                x: input placeholder.
              Returns:
                logits: Output tensor with the computed logits.
        """
        raise NotImplementedError
    # ----------------------------------------------
    def clip_grads(self, grads):
        '''
            Clip the gradients
            Args:
                grads: a list of gradients and their variables
            Returns:
                grads: the gradients after clipping 
        '''
        if not self.grad_clip:
            return grads
        with tf.name_scope('grad_clip'):
            if self.grad_clip_type == 'value':
                grads = [(tf.clip_by_value(grad, self.grad_clip_range[0], self.grad_clip_range[1]), var) 
                         for grad, var in grads] 
            elif self.grad_clip_type == 'norm':
                raise NotImplementedError
            else:
                raise NotImplementedError
        
        return grads
 
    # ----------------------------------------------
    def loss(self, logits, labels, scope=''):
        """
              Calculates the loss from the logits and the labels.
              Args:
                logits: Logits tensor, float - [batch_size, NUM_CLASSES].
                labels: Labels tensor, int32 - [batch_size].
              Returns:
                loss: Loss tensor of type float.
        """
        with tf.name_scope('loss'):
            with tf.name_scope('cross_entropy'):
                labels = tf.to_int32(labels)
                cross_entropy_mean = self.loss_func(labels=labels, logits=logits)
                tf.summary.scalar('cross_entropy', cross_entropy_mean)
                tf.add_to_collection('losses', cross_entropy_mean)
            with tf.name_scope('total_loss'):
                loss = tf.add_n(tf.get_collection('losses',scope), name='total_loss')
                tf.summary.scalar('total_loss', loss)
            return loss

    # ----------------------------------------------
    def get_optimizer(self):
        """
              Sets up the training optimizer.
              Creates an optimizer with decaying learning rate.
              Returns:
                optimizer: the optimizer for training
        """
        with tf.name_scope('learning_rate_decay'):
            learning_rate = tf.train.exponential_decay(learning_rate = self.learning_rate, 
                                                       global_step = self.global_step,
                                                       decay_steps = self.learning_rate_decay_steps, 
                                                       decay_rate = self.learning_rate_decay_rate, 
                                                       staircase=True)
            tf.summary.scalar('learning_rate', learning_rate)
            
        if self.optimizer =='sgd':
            # Create the gradient descent optimizer with the given learning rate.
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

        elif self.optimizer =='adam':
            # Create the ADAM optimizer with the given learning rate.
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1= self.adam_beta1)

        elif self.optimizer =='rmsprop':
            # Create the RMSProp optimizer with the given learning rate.
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                                  decay = self.rmsprop_decay)
        return optimizer
    # ----------------------------------------------
    @staticmethod
    def flatten(x):
        """
            flatten the input tensor if needed.
              Args:
                x: input placeholder [batch_size, ...,...,...].
              Returns:
                x: Output tensor [batch_size, num_total_features].
        """
        return model.reshape(x=x, target_dim=2)
    # ----------------------------------------------
    @staticmethod
    def reshape(x, target_dim = 3):
        """
            flatten the input tensor if needed.
              Args:
                x: input placeholder [batch_size, ...,...,...].
                target_dim: reshape to how many dims (1- vector, 2, matrix) 
              Returns:
                x: Output tensor [batch_size, num_total_features].
        """
        x_shape = x.get_shape().as_list()
        if len(x_shape) > target_dim:
            with tf.name_scope('reshape_to_%dd'% target_dim):
                n_input_dim=1
                for i in xrange(target_dim-1, len(x_shape)):
                    n_input_dim *=  x_shape[i] 
                x = tf.reshape(x,[-1]+ x_shape[1:target_dim-1]+[n_input_dim])
        elif len(x_shape) < target_dim:
            with tf.name_scope('reshape_to_%dd'% target_dim):
                dim_to_add = target_dim - len(x_shape)
                x = tf.reshape(x,[-1]+ x_shape[1:]+[1]*dim_to_add)
        return x
    # ----------------------------------------------
    def variable(self, name, shape, l2_regularization=None, initializer='norm', stddev=0.02, const = 0.0):
        """
            Create an initialized Variable.
              The Variable can be initialized with a  normal distribution or as a constant.
              A weight decay is added only if one is specified.
              Args:
                name: name of the variable
                shape: list of ints
                stddev: standard deviation of a truncated Gaussian
                l2_regularization: add L2Loss weight decay multiplied by this float. If None, weight
                    decay is not added for this Variable.
              Returns:
                Variable Tensor
          """
        if initializer=='norm':
            initializer = tf.random_normal_initializer(stddev=stddev, dtype=self.dtype)
        elif initializer=='const':
            initializer = tf.constant_initializer(const)
        else:
            initializer = initializer
        
        if self.cpu:
            with tf.device('/cpu:0'):
                var = tf.get_variable(name, shape, initializer=initializer, dtype=self.dtype)
        else:
            var = tf.get_variable(name, shape, initializer=initializer, dtype=self.dtype)
            
        if l2_regularization is not None:
            with tf.name_scope('l2_regularization'):
                weight_decay = tf.multiply(tf.nn.l2_loss(var), l2_regularization, name='l2_loss')
                tf.add_to_collection('losses', weight_decay)
        return var
    # ----------------------------------------------
    @staticmethod
    def activation(name = 'relu'):
        '''get activation function'''
        if name == 'relu': # max(features, 0)
            activation = tf.nn.relu
        elif name == 'relu6': # min(max(features, 0), 6)
            activation = tf.nn.relu6
        elif name == 'crelu': # Concatenates a ReLU which selects only the positive part of the activation with a ReLU which selects only the negative part of the activation
            activation = tf.nn.crelu
        elif name == 'elu': # exp(features) - 1 if < 0, features otherwise
            activation = tf.nn.elu
        elif name == 'sigmoid': # 1 / (1 + exp(-x))
            activation = tf.nn.sigmoid
        elif name == 'tanh': # tangent of x
            activation = tf.nn.tanh
        elif name == 'softplus': # log(exp(features) + 1)
            activation = tf.nn.softplus 
        elif name == 'softsign': # features / (abs(features) + 1)
            activation = tf.nn.softsign
        elif name == 'lrelu': # leaky ReLU
            def lrelu(X, leak=0.2,name='leaky_ReLU'):
                f1 = 0.5 * (1 + leak)
                f2 = 0.5 * (1 - leak)
                return f1 * X + f2 * tf.abs(X)
            activation = lrelu
        elif name == 'linear': # no activation function (linear function)
            def linear(x):
                return x
            activation = linear
        else: 
            raise NotImplementedError
        return activation

    # ---------------------------------------------- 
    @staticmethod
    def batchnormalize(x, variance_epsilon=1e-8, scale=None, offset=None):
        '''
            batch normalization.
            Inputs:
                x: input tensor of [batch_size, ...,...,...]
                variance_epsilon:  A small float number to avoid dividing by 0.
        '''
        with tf.variable_scope('batchnorm'):
            #global normalization
            if x.get_shape().ndims ==4:
                mean, variance = tf.nn.moments(x, axes=[0, 1, 2])
                x = tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
            # batch normalization
            elif x.get_shape().ndims == 2:
                mean, variance = tf.nn.moments(x, axes=[0])
                x = tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
            else:
                raise NotImplementedError
        return x

    # ---------------------------------------------- 
    @staticmethod
    def loss_func(logits, labels):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    # ----------------------------------------------   
    @staticmethod
    def summaries(var):
        """Attach summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)  

    # ----------------------------------------------
    def embedding_layer(self,x):
        if x.dtype is not tf.int32:
            return x
         
        print('Embedding layer is active.SOMETHING IS WRONG??? CHECK HOW THIS METHOD IS REACHED!!!')
        
        # add summary to input sequences
        with tf.name_scope('summary_input_seq'):
            input_text = self.get_text_from_ids(x)
            tf.summary.text('input', input_text)

        batch_size, max_seq_len = x.get_shape().as_list()[:2]

        n_vocab = len(self.vocab)
        with tf.name_scope('seq_embedding'):
            # Look up embeddings for inputs.
            embeddings = self.variable( name='embeddings',
                                        shape=[n_vocab, self.embedding_dim],
                                        initializer = tf.random_uniform_initializer(-1.0, 1.0),
                                        l2_regularization=self.l2_regularization)
            x_embed = tf.nn.embedding_lookup(embeddings, x)
            self.input_shape = [batch_size, max_seq_len, self.embedding_dim,1]
            x_summary = tf.reshape(x_embed,self.input_shape)
            tf.summary.image('embeddings', x_summary, max_outputs=10)

        return x_summary
    
    # ----------------------------------------------
    def attention_layer(self, inputs, attention_size, time_major=True, return_alphas=False):
        if time_major:
            # (T,B,D) => (B,T,D)
            inputs = tf.transpose(inputs, perm=[1, 0, 2])
        inputs_shape = inputs.shape
        sequence_length = inputs_shape[1].value  # the length of sequences processed in the antecedent RNN layer
        hidden_size = inputs_shape[2].value  # hidden size of the RNN layer

        # Attention mechanism
        W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1), name='W_omega')
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1), name='b_omega')
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1), name='u_omega')

        v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
        vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
        exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
        alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
        # Output of Bi-RNN is reduced with attention vector
        output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)
        if not return_alphas:
            return output
        else:
      
    
#-------------------------------------------------------
# Standard Deep Learning Models
#-------------------------------------------------------

class SoftmaxRegression(model):
    '''SoftmaxRegression is a softmax regression model with a single linear layer'''
    def __init__(self):
        super(SoftmaxRegression, self).__init__()

    def inference(self, x, n_outputs=None, bn = False, condition = None):
        """Build the model up to where it may be used for inference.
              Args:
                x: input placeholder
                n_outputs (optional): the number of output units (if it is not the number of classes)
                bn: whether or not to perform batch normalization
              Returns:
                softmax_linear: Output tensor with the computed logits.
        """
        
        print('Cansu softmax inference has called ')
        if n_outputs is None:
            n_outputs = self.n_class
            
        with tf.variable_scope('SoftmaxRegression'):
            x = self.embedding_layer(x)
            x = self.flatten(x)
            if condition is not None:
                with tf.name_scope('condition'):
                    x = tf.concat(values=[x,condition], axis = 1)
            
            n_input_dim = x.get_shape().as_list()[1]
            weights = self.variable(name='weights',
                                    shape=[n_input_dim, n_outputs], 
                                    l2_regularization=self.l2_regularization)
            
            wx = tf.matmul(x, weights)
            if bn:
                logits = self.batchnormalize(wx) 
            else:
                biases = self.variable(name='biases',
                                   shape = [n_outputs],
                                   initializer='const')
                logits = wx + biases # predicted label
              
        return logits
    
#-------------------------------------------------------
#-------------------------------------------------------
class FullyConnected(SoftmaxRegression):
    '''FullyConnected is a fully-connected neural network model with multiple layers'''
    def __init__(self,
                 fc_layers = [32], # number of nodes in each layer
                 fc_dropout_rate = 0.2, # 0 means NO dropout, 0.1 means 10% of the units will be dropped
                 fc_batchnorm = False, # whether or not to perform batch normalization
                 fc_activation = 'tanh' # activation function: tanh, sigmoid, relu, relu6, crelu, elu, ...
                ):
        super(FullyConnected, self).__init__()
        self.fc_layers = fc_layers
        self.fc_batchnorm = fc_batchnorm
        self.fc_dropout_rate = fc_dropout_rate
        self.fc_activation = fc_activation
    
    def fully_connected_layer(self,x, n_outputs, is_last_layer= False,condition = None,bn = None):
        """Build one layer of FullyConnected network.
              Args:
                x: input tensor
                n_outputs: number of hidden units 
                is_last_layer: whether this is the last layer of the model
                condition: the tensor for the layer to condition on
              Returns:
                x: Output tensor with the computed activations.
        """
        print('Fully connected layer  has called ')
        if bn is None and (not is_last_layer):
            bn = self.fc_batchnorm
        x = SoftmaxRegression.inference(self, 
                                        x, 
                                        n_outputs=n_outputs,
                                        bn = bn,
                                        condition = condition 
                                       )
        if not is_last_layer:
            with tf.name_scope('activations'):  
                activation = self.activation(self.fc_activation)
                x = activation(x) 
            if self.fc_dropout_rate >0:
                with tf.name_scope('dropout'):
                    x = tf.nn.dropout(x, 1-self.fc_dropout_rate)
        return x

        
    def inference(self, x , n_outputs = None, hidden_layers = None, condition = None):
        """Build the FullyConnected model up to where it may be used for inference.
              Args:
                x: input placeholder
                n_outputs: the number of output nodes if it is not n_class
                hidden_layers: # number of nodes in each layer, if it is not fc_layers
                condition: the tensor for all layers to condition on
              Returns:
                logits: Output tensor with the computed logits.
        """
        print('Fully connected inference has called ')
        with tf.variable_scope('FullyConnected'):

            if x.dtype is tf.int32:
                #x = self.embedding_layer(x)
                pre_trained_embeddings = np.load('datasets/ClinicalTS/BaseFiles/cdif/w2v/final_embeddings.npy')
                VOCAB_LENGTH, EMBEDDING_DIMENSION = pre_trained_embeddings.shape
                print('Embedding matrix dimensions, ',VOCAB_LENGTH, EMBEDDING_DIMENSION)
            
                embeddings = tf.get_variable(
                    name='embeddings', 
                    shape=(VOCAB_LENGTH, EMBEDDING_DIMENSION), 
                    initializer= tf.constant_initializer(pre_trained_embeddings),
                    trainable=False)

                x = tf.nn.embedding_lookup(embeddings, x)

            if hidden_layers is not None:
                n_nodes = hidden_layers
            else:
                n_nodes = self.fc_layers
            if n_outputs is not None:
                n_nodes = n_nodes + [n_outputs]
            else:
                n_nodes = n_nodes + [self.n_class]  
            
            n_layers = len(n_nodes)
            for i in xrange(n_layers):
                with tf.variable_scope('layer_%d'% i):
                    x = self.fully_connected_layer(x,
                                                   n_nodes[i], 
                                                   is_last_layer=(i==(n_layers-1)),
                                                   condition = condition
                                                  )
        return x

#-------------------------------------------------------

#-------------------------------------------------------
class RecurrentNN(FullyConnected):
    '''RNN is a recurrent neural network model '''
    def __init__(self,
                 rnn_cell_type='lstm', # blstm, rnn, gru, lstm, 
                 rnn_layers = [32], # number of cells in each layer
                 rnn_dropout_rates = [0.2,0.2,0.2], # dropout rates for [input, state, output] of RNN
                 rnn_activation = 'tanh' # activation function: tanh, sigmoid, relu, relu6, crelu, elu, ...
                ):
        super(RecurrentNN, self).__init__()
        self.rnn_cell_type= rnn_cell_type
        self.rnn_layers = rnn_layers
        self.rnn_dropout_rates = rnn_dropout_rates
        self.rnn_activation=rnn_activation
        
        # changes to the default parameter settings in parent classes
        self.fc_layers = [] # number of nodes in each layer
         
    def rnn_cell(self,n_cells):
        """Build the single-layer RNN cell
              Returns:
                cell: single-layer RNN cell
        """

        activation = self.activation(name=self.rnn_activation)
        with tf.device('/cpu:0'):
            with tf.name_scope('rnn_cell'):
                if self.rnn_cell_type == 'blstm':
                    rnn_cell = tf.contrib.rnn.BasicLSTMCell(n_cells,activation=activation)
                elif self.rnn_cell_type == 'rnn':
                    rnn_cell = tf.contrib.rnn.BasicRNNCell(n_cells,activation=activation)
                elif self.rnn_cell_type == 'gru':
                    rnn_cell = tf.contrib.rnn.GRUCell(n_cells,activation=activation)
                elif self.rnn_cell_type == 'lstm':
                    rnn_cell = tf.contrib.rnn.LSTMCell(n_cells,activation=activation)
                else: 
                    raise NotImplementedError
        
        if sum(self.rnn_dropout_rates)>0:
            with tf.name_scope('dropout'):
                rnn_cell= tf.contrib.rnn.DropoutWrapper(rnn_cell,
                                                        input_keep_prob=1-self.rnn_dropout_rates[0],
                                                        output_keep_prob=1-self.rnn_dropout_rates[2],
                                                        state_keep_prob=1-self.rnn_dropout_rates[1])
        return rnn_cell
    
    def multi_rnn_cell(self,batch_size):
        """Build the Multi-layer RNN cell and initial state
                batch_size: the size of a batch
              Returns:
                cell: multi-layer RNN cell
                state: initial state
        """
        rnn_cell = tf.contrib.rnn.MultiRNNCell([self.rnn_cell(n_cells) for n_cells in self.rnn_layers])
        state = rnn_cell.zero_state(batch_size,dtype=self.dtype)
        return rnn_cell, state

    def inference(self, x):
        """Build the RNN model up to where it may be used for inference.
              Args:
                x: input placeholder
              Returns:
                logits: Output tensor with the computed logits.
        """
        
        with tf.variable_scope('RecurrentNN'):
        
            pre_trained_embeddings = np.load('datasets/ClinicalTS/BaseFiles/cdif/w2v/final_embeddings.npy')
            VOCAB_LENGTH, EMBEDDING_DIMENSION = pre_trained_embeddings.shape
            print('Embedding matrix dimensions, ',VOCAB_LENGTH, EMBEDDING_DIMENSION)
            
            ## Old code: embeddings = tf.Variable(pre_trained_embeddings, trainable=False)
            embeddings = tf.get_variable(
                name='embeddings', 
                shape=(VOCAB_LENGTH, EMBEDDING_DIMENSION), 
                initializer= tf.constant_initializer(pre_trained_embeddings),
                trainable=False)

            if x.dtype is tf.int32:
                batch_size, n_steps  = x.get_shape().as_list()
            else:
                batch_size, n_steps, _  = x.get_shape().as_list()

            with tf.name_scope('unstack'):
                x_seq = tf.unstack(x, n_steps, 1)

            rnn_cell, state = self.multi_rnn_cell(batch_size)
            for i in xrange(n_steps):
                if x.dtype is tf.int32:
                    x_seq[i] = tf.nn.embedding_lookup(embeddings, x_seq[i])
                output, state = rnn_cell(x_seq[i], state)
            logits = FullyConnected.inference(self,output)       
        return logits

#---------------------------------------------------- 
class RecurrentNN_wAtt(FullyConnected):
    '''RNN is a recurrent neural network model '''
    def __init__(self,
                 rnn_cell_type='lstm', # blstm, rnn, gru, lstm, 
                 rnn_layers = [32], # number of cells in each layer
                 rnn_dropout_rates = [0.2,0.2,0.2], # dropout rates for [input, state, output] of RNN
                 rnn_activation = 'tanh' # activation function: tanh, sigmoid, relu, relu6, crelu, elu, ...
                ):
        super(RecurrentNN_wAtt, self).__init__()
        self.rnn_cell_type= rnn_cell_type
        self.rnn_layers = rnn_layers
        self.rnn_dropout_rates = rnn_dropout_rates
        self.rnn_activation=rnn_activation
        
        # changes to the default parameter settings in parent classes
        self.fc_layers = [] # number of nodes in each layer
    def rnn_cell(self,n_cells):
        """Build the single-layer RNN cell
              Returns:
                cell: single-layer RNN cell
        """

        activation = self.activation(name=self.rnn_activation)
        with tf.device('/cpu:0'):
            with tf.name_scope('rnn_cell'):
                if self.rnn_cell_type == 'blstm':
                    rnn_cell = tf.contrib.rnn.BasicLSTMCell(n_cells,activation=activation)
                elif self.rnn_cell_type == 'rnn':
                    rnn_cell = tf.contrib.rnn.BasicRNNCell(n_cells,activation=activation)
                elif self.rnn_cell_type == 'gru':
                    rnn_cell = tf.contrib.rnn.GRUCell(n_cells,activation=activation)
                elif self.rnn_cell_type == 'lstm':
                    rnn_cell = tf.contrib.rnn.LSTMCell(n_cells,activation=activation)
                else: 
                    raise NotImplementedError
        
        if sum(self.rnn_dropout_rates)>0:
            with tf.name_scope('dropout'):
                rnn_cell= tf.contrib.rnn.DropoutWrapper(rnn_cell,
                                                        input_keep_prob=1-self.rnn_dropout_rates[0],
                                                        output_keep_prob=1-self.rnn_dropout_rates[2],
                                                        state_keep_prob=1-self.rnn_dropout_rates[1])
        return rnn_cell
    
    def multi_rnn_cell(self,batch_size):
        """Build the Multi-layer RNN cell and initial state
                batch_size: the size of a batch
              Returns:
                cell: multi-layer RNN cell
                state: initial state
        """
        rnn_cell = tf.contrib.rnn.MultiRNNCell([self.rnn_cell(n_cells) for n_cells in self.rnn_layers])
        state = rnn_cell.zero_state(batch_size,dtype=self.dtype)
        return rnn_cell, state

    def inference(self, x):
        with tf.variable_scope('RecurrentNN_wAtt'):
            
            withAttention = True
            
            pre_trained_embeddings = np.load('datasets/ClinicalTS/BaseFiles/mort/w2v/final_embeddings.npy')
            VOCAB_LENGTH, EMBEDDING_DIMENSION = pre_trained_embeddings.shape
            print('Embedding matrix dimensions, ',VOCAB_LENGTH, EMBEDDING_DIMENSION)
        
            embeddings = tf.get_variable(name='embeddings', shape=(VOCAB_LENGTH, EMBEDDING_DIMENSION), 
                                        initializer= tf.constant_initializer(pre_trained_embeddings),trainable=False)

            if x.dtype is tf.int32:
                batch_size, n_steps  = x.get_shape().as_list()
            else:
                batch_size, n_steps, _  = x.get_shape().as_list()

            with tf.name_scope('unstack'):
                x_seq = tf.unstack(x, n_steps, 1)
            
            rnn_cell, state = self.multi_rnn_cell(batch_size)
            stacked_outputs = []
            for i in xrange(n_steps):
                if x.dtype is tf.int32:
                    print('Word input, using pre-trained embeddings')
                    x_seq[i] = tf.nn.embedding_lookup(embeddings, x_seq[i])
                output, state = rnn_cell(x_seq[i], state)
                stacked_outputs.append(output)

            stacked_outputs = tf.stack(stacked_outputs)

            # Attention layer
            if withAttention:
                ATTENTION_SIZE = 32
                attention_output, alphas = self.attention_layer(stacked_outputs, ATTENTION_SIZE, return_alphas=True)
                #tf.summary.histogram("att_weights", W_omega)
                #tf.summary.histogram('att_v', v)
                for i in range(alphas.get_shape()[0]):
                    for j in range(alphas.get_shape()[1]):
                        tf.summary.scalar('alphas' + str(i) + '_' + str(j), alphas[i, j])
                #tf.summary.tensor_summary('alphas', alphas)
                #print('attention out', attention_output)
                logits = FullyConnected.inference(self,attention_output) 
            else:
                #print('output', output)
                logits = FullyConnected.inference(self,output) 
                  
        return logits

#-------------------------------------------------------
# Hieracrhical Recurrent Neurel Network 
#-------------------------------------------------------
class Hierarchical_RNN(FullyConnected):
    '''Hierarchical_RNN is a recurrent neural network model '''
    def __init__(self,
                 rnn_cell_type='blstm', # blstm, rnn, gru, lstm, 
                 rnn_layers = [8], # number of cells in each layer
                 rnn_dropout_rates = [0.2,0.2,0.2], # dropout rates for [input, state, output] of RNN
                 rnn_activation = 'tanh' # activation function: tanh, sigmoid, relu, relu6, crelu, elu, ...
                ):
        super(Hierarchical_RNN, self).__init__()
        self.rnn_cell_type= rnn_cell_type
        self.rnn_layers = rnn_layers
        self.rnn_dropout_rates = rnn_dropout_rates
        self.rnn_activation=rnn_activation
        
        # changes to the default parameter settings in parent classes
        self.fc_layers = [] # number of nodes in each layer
        
    
    def rnn_cell(self,n_cells):
        """Build the single-layer RNN cell
              Returns:
                cell: single-layer RNN cell
        """

        activation = self.activation(name=self.rnn_activation)
        with tf.device('/cpu:0'):
            with tf.name_scope('rnn_cell'):
                if self.rnn_cell_type == 'blstm':
                    rnn_cell = tf.contrib.rnn.BasicLSTMCell(n_cells,activation=activation)
                elif self.rnn_cell_type == 'rnn':
                    rnn_cell = tf.contrib.rnn.BasicRNNCell(n_cells,activation=activation)
                elif self.rnn_cell_type == 'gru':
                    rnn_cell = tf.contrib.rnn.GRUCell(n_cells,activation=activation)
                elif self.rnn_cell_type == 'lstm':
                    rnn_cell = tf.contrib.rnn.LSTMCell(n_cells,activation=activation)
                else: 
                    raise NotImplementedError
        
        if sum(self.rnn_dropout_rates)>0:
            with tf.name_scope('dropout'):
                rnn_cell= tf.contrib.rnn.DropoutWrapper(rnn_cell,
                                                        input_keep_prob=1-self.rnn_dropout_rates[0],
                                                        output_keep_prob=1-self.rnn_dropout_rates[2],
                                                        state_keep_prob=1-self.rnn_dropout_rates[1])
        return rnn_cell
    
    def multi_rnn_cell(self,batch_size):
        """Build the Multi-layer RNN cell and initial state
                batch_size: the size of a batch
              Returns:
                cell: multi-layer RNN cell
                state: initial state
        """
        rnn_cell = tf.contrib.rnn.MultiRNNCell([self.rnn_cell(n_cells) for n_cells in self.rnn_layers])
        state = rnn_cell.zero_state(batch_size,dtype=self.dtype)
        return rnn_cell, state

    def inference(self, x):
        """Build the RNN model up to where it may be used for inference.
              Args:
                x: input placeholder
              Returns:
                logits: Output tensor with the computed logits.
        """
        #n_vocab = len(self.vocab)
        
        #embeddings = self.variable( name='embeddings',
        #                            shape=[n_vocab, self.embedding_dim],
        #                            initializer = tf.random_uniform_initializer(-1.0, 1.0),
        #                            l2_regularization=self.l2_regularization)
        pre_trained_embeddings = np.load('datasets/ClinicalTS/BaseFiles/mrsa/w2v/final_embeddings.npy')
        VOCAB_LENGTH, EMBEDDING_DIMENSION = pre_trained_embeddings.shape
        print('Embedding matrix dimensions, ',VOCAB_LENGTH, EMBEDDING_DIMENSION)
        
        ## Old code: embeddings = tf.Variable(pre_trained_embeddings, trainable=False)
        embeddings = tf.get_variable(
            name='embeddings', 
            shape=(VOCAB_LENGTH, EMBEDDING_DIMENSION), 
            initializer= tf.constant_initializer(pre_trained_embeddings),
            trainable=False)
        
        #n_steps documents, each corresponding to a time step
        #n_words words in each document
        batch_size, n_steps, n_words = x.get_shape().as_list()
        print('batch_size', batch_size, 'n_steps', n_steps, 'n_words',n_words)

        with tf.name_scope('unstack'):
            x_seq = tf.unstack(x, n_steps, 1)
        
        with tf.variable_scope('Hierarchical_RNN'):
            rnn_cell, state = self.multi_rnn_cell(batch_size)
            docu_states = []
            for i in xrange(n_steps):
                for j in xrange(n_words): 
                    word_input = tf.slice(x_seq[i], [0,j], [batch_size,1]) 
                    word_input = tf.reshape(word_input, [batch_size])
                    word_input = tf.transpose(word_input)
                    word_embed = tf.nn.embedding_lookup(embeddings, word_input)
                    output, state = rnn_cell(word_embed, state)
                docu_states.append(output)

        docu_states = tf.stack(docu_states)
        #docu_states = tf.transpose(docu_states, perm=[1, 0, 2])
        #print('Combined output of word level:', docu_states)
        with tf.variable_scope('Hierarchical_RNN_2'):
            # input concated states from the previous layer: docu_states
            rnn_cell, state = self.multi_rnn_cell(batch_size)
            for i in xrange(n_steps):
                output, state = rnn_cell(docu_states[i], state)
            
            # Final output is calculated here
            logits = FullyConnected.inference(self,output)   
   
        return logits

#-------------------------------------------------------
# Hieracrhical Recurrent Neurel Network with Attention
#-------------------------------------------------------
class Hierarchical_RNN_wAtt(FullyConnected):
    '''Hierarchical_RNN is a recurrent neural network model '''
    def __init__(self,
                 rnn_cell_type='lstm', # blstm, rnn, gru, lstm, 
                 rnn_layers = [8], # number of cells in each layer
                 rnn_dropout_rates = [0.2,0.2,0.2], # dropout rates for [input, state, output] of RNN
                 rnn_activation = 'tanh' # activation function: tanh, sigmoid, relu, relu6, crelu, elu, ...
                ):
        super(Hierarchical_RNN_wAtt, self).__init__()
        self.rnn_cell_type= rnn_cell_type
        self.rnn_layers = rnn_layers
        self.rnn_dropout_rates = rnn_dropout_rates
        self.rnn_activation=rnn_activation
        
        # changes to the default parameter settings in parent classes
        self.fc_layers = [] # number of nodes in each layer
        
    
    def rnn_cell(self,n_cells):
        """Build the single-layer RNN cell
              Returns:
                cell: single-layer RNN cell
        """

        activation = self.activation(name=self.rnn_activation)
        with tf.device('/cpu:0'):
            with tf.name_scope('rnn_cell'):
                if self.rnn_cell_type == 'blstm':
                    rnn_cell = tf.contrib.rnn.BasicLSTMCell(n_cells,activation=activation)
                elif self.rnn_cell_type == 'rnn':
                    rnn_cell = tf.contrib.rnn.BasicRNNCell(n_cells,activation=activation)
                elif self.rnn_cell_type == 'gru':
                    rnn_cell = tf.contrib.rnn.GRUCell(n_cells,activation=activation)
                elif self.rnn_cell_type == 'lstm':
                    rnn_cell = tf.contrib.rnn.LSTMCell(n_cells,activation=activation)
                else: 
                    raise NotImplementedError
        
        if sum(self.rnn_dropout_rates)>0:
            with tf.name_scope('dropout'):
                rnn_cell= tf.contrib.rnn.DropoutWrapper(rnn_cell,
                                                        input_keep_prob=1-self.rnn_dropout_rates[0],
                                                        output_keep_prob=1-self.rnn_dropout_rates[2],
                                                        state_keep_prob=1-self.rnn_dropout_rates[1])
        return rnn_cell
    
    def multi_rnn_cell(self,batch_size):
        """Build the Multi-layer RNN cell and initial state
                batch_size: the size of a batch
              Returns:
                cell: multi-layer RNN cell
                state: initial state
        """
        rnn_cell = tf.contrib.rnn.MultiRNNCell([self.rnn_cell(n_cells) for n_cells in self.rnn_layers])
        state = rnn_cell.zero_state(batch_size,dtype=self.dtype)
        return rnn_cell, state

    def inference(self, x):
        """Build the RNN model up to where it may be used for inference.
              Args:
                x: input placeholder
              Returns:
                logits: Output tensor with the computed logits.
        """
        
        pre_trained_embeddings = np.load('datasets/ClinicalTS/BaseFiles/cdif/w2v/final_embeddings.npy')
        VOCAB_LENGTH, EMBEDDING_DIMENSION = pre_trained_embeddings.shape
        print('Embedding matrix dimensions, ',VOCAB_LENGTH, EMBEDDING_DIMENSION)
        
        ## Old code: embeddings = tf.Variable(pre_trained_embeddings, trainable=False)
        embeddings = tf.get_variable(
            name='embeddings', 
            shape=(VOCAB_LENGTH, EMBEDDING_DIMENSION), 
            initializer= tf.constant_initializer(pre_trained_embeddings),
            trainable=False)
        
        #n_steps documents, each corresponding to a time step
        #n_words words in each document
        batch_size, n_steps, n_words = x.get_shape().as_list()
        print('batch_size', batch_size, 'n_steps', n_steps, 'n_words',n_words)

        with tf.name_scope('unstack'):
            x_seq = tf.unstack(x, n_steps, 1)
        
        with tf.variable_scope('Hierarchical_RNN_wAtt'):
            rnn_cell, state = self.multi_rnn_cell(batch_size)
            docu_states = []
            for i in xrange(n_steps):
                stacked_outputs = []
                for j in xrange(n_words): 
                    word_input = tf.slice(x_seq[i], [0,j], [batch_size,1]) 
                    word_input = tf.reshape(word_input, [batch_size])
                    word_input = tf.transpose(word_input)
                    word_embed = tf.nn.embedding_lookup(embeddings, word_input)
                    output, state = rnn_cell(word_embed, state)
                    stacked_outputs.append(output) #Word level outputs to pass the attention
                stacked_outputs = tf.stack(stacked_outputs)
                ATTENTION_SIZE = 16
                attention_output, alphas = self.attention_layer(stacked_outputs, ATTENTION_SIZE, return_alphas=True)
                docu_states.append(attention_output) #Only last outputs to pass the second level

        docu_states = tf.stack(docu_states)
        #docu_states = tf.transpose(docu_states, perm=[1, 0, 2])
        #print('Combined output of word level:', docu_states)
        with tf.variable_scope('Hierarchical_RNN_wAtt_2'):
            # input concated states from the previous layer: docu_states
            rnn_cell, state = self.multi_rnn_cell(batch_size)
            for i in xrange(n_steps):
                output, state = rnn_cell(docu_states[i], state)
            
            # Final output is calculated here
            logits = FullyConnected.inference(self,output)   
   
        return logits

#-------------------------------------------------------
# Hieracrhical Recurrent Neurel Network with Double Attention
#-------------------------------------------------------
class Hierarchical_RNN_wDoubleAtt(FullyConnected):
    '''Hierarchical_RNN is a recurrent neural network model '''
    def __init__(self,
                 rnn_cell_type='gru', # blstm, rnn, gru, lstm, 
                 rnn_layers = [64], # number of cells in each layer
                 rnn_dropout_rates = [0.2,0.2,0.2], # dropout rates for [input, state, output] of RNN
                 rnn_activation = 'tanh' # activation function: tanh, sigmoid, relu, relu6, crelu, elu, ...
                ):
        super(Hierarchical_RNN_wDoubleAtt, self).__init__()
        self.rnn_cell_type= rnn_cell_type
        self.rnn_layers = rnn_layers
        self.rnn_dropout_rates = rnn_dropout_rates
        self.rnn_activation=rnn_activation
        
        # changes to the default parameter settings in parent classes
        self.fc_layers = [] # number of nodes in each layer
        
    
    def rnn_cell(self,n_cells):
        """Build the single-layer RNN cell
              Returns:
                cell: single-layer RNN cell
        """

        activation = self.activation(name=self.rnn_activation)
        with tf.device('/cpu:0'):
            with tf.name_scope('rnn_cell'):
                if self.rnn_cell_type == 'blstm':
                    rnn_cell = tf.contrib.rnn.BasicLSTMCell(n_cells,activation=activation)
                elif self.rnn_cell_type == 'rnn':
                    rnn_cell = tf.contrib.rnn.BasicRNNCell(n_cells,activation=activation)
                elif self.rnn_cell_type == 'gru':
                    rnn_cell = tf.contrib.rnn.GRUCell(n_cells,activation=activation)
                elif self.rnn_cell_type == 'lstm':
                    rnn_cell = tf.contrib.rnn.LSTMCell(n_cells,activation=activation)
                else: 
                    raise NotImplementedError
        
        if sum(self.rnn_dropout_rates)>0:
            with tf.name_scope('dropout'):
                rnn_cell= tf.contrib.rnn.DropoutWrapper(rnn_cell,
                                                        input_keep_prob=1-self.rnn_dropout_rates[0],
                                                        output_keep_prob=1-self.rnn_dropout_rates[2],
                                                        state_keep_prob=1-self.rnn_dropout_rates[1])
        return rnn_cell
    
    def multi_rnn_cell(self,batch_size):
        """Build the Multi-layer RNN cell and initial state
                batch_size: the size of a batch
              Returns:
                cell: multi-layer RNN cell
                state: initial state
        """
        rnn_cell = tf.contrib.rnn.MultiRNNCell([self.rnn_cell(n_cells) for n_cells in self.rnn_layers])
        state = rnn_cell.zero_state(batch_size,dtype=self.dtype)
        return rnn_cell, state

    def inference(self, x):
        """Build the RNN model up to where it may be used for inference.
              Args:
                x: input placeholder
              Returns:
                logits: Output tensor with the computed logits.
        """
        
        pre_trained_embeddings = np.load('datasets/ClinicalTS/BaseFiles/mort/w2v/final_embeddings.npy')
        VOCAB_LENGTH, EMBEDDING_DIMENSION = pre_trained_embeddings.shape
        print('Embedding matrix dimensions, ',VOCAB_LENGTH, EMBEDDING_DIMENSION)
        
        embeddings = tf.get_variable(
            name='embeddings', 
            shape=(VOCAB_LENGTH, EMBEDDING_DIMENSION), 
            initializer= tf.constant_initializer(pre_trained_embeddings),
            trainable=True)
        
        #n_steps documents, each corresponding to a time step
        #n_words words in each document
        batch_size, n_steps, n_words = x.get_shape().as_list()
        print('batch_size', batch_size, 'n_steps', n_steps, 'n_words',n_words)

        with tf.name_scope('unstack'):
            x_seq = tf.unstack(x, n_steps, 1)
        
        with tf.variable_scope('Hierarchical_RNN_wDoubleAtt'):
            rnn_cell, state = self.multi_rnn_cell(batch_size)
            docu_states = []
            for i in xrange(n_steps):
                stacked_outputs = []
                for j in xrange(n_words): 
                    word_input = tf.slice(x_seq[i], [0,j], [batch_size,1]) 
                    word_input = tf.reshape(word_input, [batch_size])
                    word_input = tf.transpose(word_input)
                    word_embed = tf.nn.embedding_lookup(embeddings, word_input)
                    output, state = rnn_cell(word_embed, state)
                    stacked_outputs.append(output) #Word level outputs to pass the attention
                stacked_outputs = tf.stack(stacked_outputs)
                ATTENTION_SIZE = 64
                attention_output, alphas = self.attention_layer(stacked_outputs, ATTENTION_SIZE, return_alphas=True)
                #for i in range(alphas.get_shape()[0]):
                #    for j in range(alphas.get_shape()[1]):
                #        tf.summary.scalar('alphas_word' + str(i) + '_' + str(j), alphas[i, j])

                docu_states.append(attention_output) #Only last outputs to pass the second level

        docu_states = tf.stack(docu_states)
        with tf.variable_scope('Hierarchical_RNN_wDoubleAtt_2'):
            # input concated states from the previous layer: docu_states
            rnn_cell, state = self.multi_rnn_cell(batch_size)
            n_stacked_outputs = []
            for i in xrange(n_steps):
                output, state = rnn_cell(docu_states[i], state)
                n_stacked_outputs.append(output)
            n_stacked_outputs = tf.stack(n_stacked_outputs)
            
            # Final output is calculated here
            ATTENTION_SIZE = 64
            attention_output, alphas = self.attention_layer(n_stacked_outputs, ATTENTION_SIZE, return_alphas=True)
            #for i in range(alphas.get_shape()[0]):
            #    for j in range(alphas.get_shape()[1]):
            #        tf.summary.scalar('alphas_note' + str(i) + '_' + str(j), alphas[i, j])

            logits = FullyConnected.inference(self,attention_output)
   
        return logits

#-------------------------------------------------------
# Hierarchical_time_RNN_wDoubleAtt - Extra featuers on the note level
#-------------------------------------------------------
class Hierarchical_time_RNN_wDoubleAtt(FullyConnected):
    '''Hierarchical_RNN is a recurrent neural network model '''
    def __init__(self,
                 rnn_cell_type='lstm', # blstm, rnn, gru, lstm, 
                 rnn_layers = [8], # number of cells in each layer
                 rnn_dropout_rates = [0.2,0.2,0.2], # dropout rates for [input, state, output] of RNN
                 rnn_activation = 'tanh' # activation function: tanh, sigmoid, relu, relu6, crelu, elu, ...
                ):
        super(Hierarchical_time_RNN_wDoubleAtt, self).__init__()
        self.rnn_cell_type= rnn_cell_type
        self.rnn_layers = rnn_layers
        self.rnn_dropout_rates = rnn_dropout_rates
        self.rnn_activation=rnn_activation
        
        # changes to the default parameter settings in parent classes
        self.fc_layers = [] # number of nodes in each layer
        
    
    def rnn_cell(self,n_cells):
        """Build the single-layer RNN cell
              Returns:
                cell: single-layer RNN cell
        """

        activation = self.activation(name=self.rnn_activation)
        with tf.device('/cpu:0'):
            with tf.name_scope('rnn_cell'):
                if self.rnn_cell_type == 'blstm':
                    rnn_cell = tf.contrib.rnn.BasicLSTMCell(n_cells,activation=activation)
                elif self.rnn_cell_type == 'rnn':
                    rnn_cell = tf.contrib.rnn.BasicRNNCell(n_cells,activation=activation)
                elif self.rnn_cell_type == 'gru':
                    rnn_cell = tf.contrib.rnn.GRUCell(n_cells,activation=activation)
                elif self.rnn_cell_type == 'lstm':
                    rnn_cell = tf.contrib.rnn.LSTMCell(n_cells,activation=activation)
                else: 
                    raise NotImplementedError
        
        if sum(self.rnn_dropout_rates)>0:
            with tf.name_scope('dropout'):
                rnn_cell= tf.contrib.rnn.DropoutWrapper(rnn_cell,
                                                        input_keep_prob=1-self.rnn_dropout_rates[0],
                                                        output_keep_prob=1-self.rnn_dropout_rates[2],
                                                        state_keep_prob=1-self.rnn_dropout_rates[1])
        return rnn_cell
    
    def multi_rnn_cell(self,batch_size):
        """Build the Multi-layer RNN cell and initial state
                batch_size: the size of a batch
              Returns:
                cell: multi-layer RNN cell
                state: initial state
        """
        rnn_cell = tf.contrib.rnn.MultiRNNCell([self.rnn_cell(n_cells) for n_cells in self.rnn_layers])
        state = rnn_cell.zero_state(batch_size,dtype=self.dtype)
        return rnn_cell, state

    def inference(self, x):
        """Build the RNN model up to where it may be used for inference.
              Args:
                x: input placeholder
              Returns:
                logits: Output tensor with the computed logits.
        """
        
        pre_trained_embeddings = np.load('datasets/ClinicalTS/BaseFiles/cdif/w2v/final_embeddings.npy')
        VOCAB_LENGTH, EMBEDDING_DIMENSION = pre_trained_embeddings.shape
        print('Embedding matrix dimensions, ',VOCAB_LENGTH, EMBEDDING_DIMENSION)
        
        embeddings = tf.get_variable(
            name='embeddings', 
            shape=(VOCAB_LENGTH, EMBEDDING_DIMENSION), 
            initializer= tf.constant_initializer(pre_trained_embeddings),
            trainable=True)
        
        #n_steps documents, each corresponding to a time step, n_words words in each document
        batch_size, n_steps, n_words = x.get_shape().as_list()
        print('batch_size', batch_size, 'n_steps', n_steps, 'n_words',n_words)
        ATTENTION_SIZE = 16

        with tf.name_scope('unstack'):
            x_seq = tf.unstack(x, n_steps, 1)
        
        ### FIRST LAYER RNN

        with tf.variable_scope('Hierarchical_time_RNN_wDoubleAtt'):
            rnn_cell, state = self.multi_rnn_cell(batch_size)
            
            docu_states  = [] #Keeps document-level representations including time [20 notes, 14 (8 rnn + 6 time)]
            stacked_time = [] #Keeps note level time representation [20 notes, 6 time]
            intermediate_logits = [] 

            for i in xrange(n_steps-1):
                print('------------------', i, '-------------------')
                print('x_seq[i]:', x_seq[i])
                stacked_outputs = []  #Word level output, only used in the first layer
                for j in xrange(300): #300 is hard-coded, fix later
                    word_input = tf.slice(x_seq[i], [0,j], [batch_size,1]) 
                    word_input = tf.reshape(word_input, [batch_size])
                    word_input = tf.transpose(word_input)
                    word_input = tf.cast(word_input, tf.int32)
                    word_embed = tf.nn.embedding_lookup(embeddings, word_input)
                    output, state = rnn_cell(word_embed, state)
                    stacked_outputs.append(output)  #Word level outputs to pass on the attention layer
                stacked_outputs = tf.stack(stacked_outputs)
                
                #Word level attention
                attention_output, alphas = self.attention_layer(stacked_outputs, ATTENTION_SIZE, return_alphas=True)
                #for i in range(alphas.get_shape()[0]):
                #    for j in range(alphas.get_shape()[1]):
                #        tf.summary.scalar('alphas_word' + str(i) + '_' + str(j), alphas[i, j])
               
                #logits_temp  = FullyConnected.inference(self,attention_output)
                #intermediate_logits.append(logits_temp)
 
                #Prep for next layer. Handle note-level things here. This part loops for each note, 20 times
                time_part = tf.slice(x_seq[i], [0,300], [batch_size,21])
                stacked_time.append(time_part)  
                concat_output = tf.concat([attention_output,time_part], 1)
                docu_states.append(concat_output) 
                print('attention output:', attention_output)
                print('time_part:', time_part)
                print('concat_output:',concat_output)
        
        
        docu_states  = tf.stack(docu_states)
        stacked_time = tf.stack(stacked_time)
        age_part = tf.slice(x_seq[20], [0,0], [batch_size,1])
        print('age_part', age_part)

        ### SECOND LAYER RNN

        with tf.variable_scope('Hierarchical_time_RNN_wDoubleAtt_2'):
            rnn_cell, state = self.multi_rnn_cell(batch_size)
            n_stacked_outputs = []
            for i in xrange(n_steps-1):
                output, state = rnn_cell(docu_states[i], state)
                n_stacked_outputs.append(output)
            n_stacked_outputs = tf.stack(n_stacked_outputs)
            
            # Note level attention
            attention_output = self.attention_layer(n_stacked_outputs, ATTENTION_SIZE, return_alphas=False)
            #attention_output = self.time_mlp_attention_layer(n_stacked_outputs, stacked_time, ATTENTION_SIZE)
            #for i in range(alphas.get_shape()[0]):
            #    for j in range(alphas.get_shape()[1]):
            #        tf.summary.scalar('alphas_note' + str(i) + '_' + str(j), alphas[i, j])
            print('second level')
            print('bir input ornek:', docu_states[0])
            print('attention_output:', attention_output)
            concat_output = tf.concat([attention_output,age_part], 1)
            print('concat of att and age:', concat_output) 
            
            logits = FullyConnected.inference(self,concat_output)
   
        return logits