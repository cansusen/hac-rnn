from abc import ABCMeta, abstractmethod
from dataset import TextData
import tensorflow as tf
import os
from tensorflow.python.client import device_lib

#-------------------------------------------------------------------------
'''
    This file includes a collection of experiment setting classes. 
    An experiment setting class should inherit the abstract class `setting`. 
'''
#-------------------------------------------------------------------------
# Abstract Class for Experiment Settings
#-------------------------------------------------------------------------
class setting(object):
    '''
        setting is an abstract class for experiment settings.
        It defines the basic APIs for experiment setting classes. 
        1. build_graph() - Builds the computational graph for running experiment.
        2. run() - run the computational graph.
    '''
    __metaclass__ = ABCMeta
    # ----------------------------------------------
    def __init__(self,
                 batch_size = 64, #16,  # number of samples in each mini-batch
                 max_steps = 50000, #500000, #1000000, # max training steps
                 learning_rate_decay_times= 20, # number of times to decay the learning rate during training
                 interval_log = 10, #100, # how many steps to save the log (results) once
                 interval_checkpoint = 500, # how many steps to save model checkpoints once
                 max_checkpoints_to_keep = 5, # how many model checkpoints to keep
                 load_checkpoint = True, # whether or not to load the latest model checkpoint before training
                 num_GPUs = 0, # the number of GPUs to be used for training
                 plot_grads = False  # whether or not to plot the gradients in tensorboard (slower if the value is True) 
                ):
        self._batch_size = batch_size
        self._max_steps = max_steps
        self._learning_rate_decay_times = learning_rate_decay_times
        self._interval_log = interval_log
        self._interval_checkpoint = interval_checkpoint
        self._max_checkpoints_to_keep =max_checkpoints_to_keep
        self._load_checkpoint = load_checkpoint
        self._plot_grads = plot_grads
        self.num_GPUs = num_GPUs
        
        self.model = None
        self.dataset = None
        self.metrics = None
    # ----------------------------------------------
    @classmethod
    def class_name(cls):
        return cls.__name__
    @property
    def name(self):
        return self.class_name()

    @property
    def batch_size(self):
        return self._batch_size
    @batch_size.setter
    def batch_size(self,value):
        self._batch_size = value
        
    @property
    def max_steps(self):
        return self._max_steps
    @max_steps.setter
    def max_steps(self,value):
        self._max_steps =  value
        
    @property
    def learning_rate_decay_times(self):
        return self._learning_rate_decay_times
    @learning_rate_decay_times.setter
    def learning_rate_decay_times(self,value):
        self._learning_rate_decay_times = value
        
    @property
    def interval_log(self):
        return self._interval_log
    @interval_log.setter
    def interval_log(self,value):
        self._interval_log = value
    
    @property
    def interval_checkpoint(self):
        return self._interval_checkpoint
    @interval_checkpoint.setter
    def interval_checkpoint(self,value):
        self._interval_checkpoint = value

    @property
    def load_checkpoint(self):
        return self._load_checkpoint
    @load_checkpoint.setter
    def load_checkpoint(self,value):
        self._load_checkpoint = value

    @property
    def num_GPUs(self):
        return self._num_GPUs
    @num_GPUs.setter
    def num_GPUs(self,value):
        # check the number of GPUs available
        local_device_protos = device_lib.list_local_devices()
        list_gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
        n_gpus = len(list_gpus)
        value = max(value, 0)
        self._num_GPUs = min(value,n_gpus)
        
    # ----------------------------------------------
    def setup(self, 
              dataset, # the dataset object of the experiment
              model,  # the deep learning model object in the experiment
              metrics=None, # the list of evaluation metrics
              log_dir='log_long/', # the path for saving experiment results 
              checkpoint_dir= 'checkpoints/' # the path for saving model checkpoints
             ):
        '''setup everything before running experiment'''
        # setup model
        model._n_class = dataset.n_class
        model._batch_size = self._batch_size
        model._learning_rate_decay_steps= self._max_steps/self._learning_rate_decay_times
        self.model=model

        # if the only 1 GPU, place variables on GPU
        if self.num_GPUs < 2:
            self.model.cpu = False
        
        # setup dataset
        self.dataset=dataset
        # setup metrics
        self.metrics=metrics
        # setup log and checkpoint directories
        self._log_dir = log_dir
        self._checkpoint_dir = checkpoint_dir
        
        # if text dataset, assign vocabulary to model
        #if isinstance(dataset,TextData):
        #self.model.vocab = self.dataset.vocab
        #self.model.char_based = self.dataset.char_based

        # reset default graph
        tf.reset_default_graph()
    
    # ----------------------------------------------
    def multi_gpu_grads(self, batch_queue, grads_fn):
        '''
            Compute gradients using single or multiple GPUs.
            In multi-GPU setting: The model parameters are in the CPU memory. 
            Each GPU has a copy of the model, and computes the gradients.
            The gradients across multiple GPUs are then averaged and applied to the model in CPU.
            Args:
                batch_queue: the tensor of queued batch of dataset
                grad_fn: the function to compute gradients of the model
                        the grad_fn should take two parameters: batch_queue, a string of name_scope
            Returns:
                grads: the final gradients
        '''
        if self.num_GPUs < 2:
            # single GPU training
            grads = grads_fn(batch_queue,'tower_0')
        else:    
            # multi-GPU training
            tower_grads = []
            for i in xrange(self.num_GPUs):
                with tf.device('/gpu:%d' % i):
                    grads = grads_fn(batch_queue,'tower_%d'% i)
                    tf.get_variable_scope().reuse_variables()
                    tower_grads.append(grads)
            grads = self.average_gradients(tower_grads)

        # gradient clipping
        grads = self.model.clip_grads(grads)

        # add gradients to summary
        if self._plot_grads:
            for g, v in grads:
                tf.summary.histogram(v.name + '/gradient', g)

        return grads 

    # ----------------------------------------------
    @staticmethod
    def average_gradients(tower_grads):
        '''
            Calculate the average gradient for each shared variable across all towers.
            Args:
                tower_grads: List of lists of (gradient, variable) tuples. The outer list
                  is over individual gradients. The inner list is over the gradient
                  calculation for each tower.
            Returns:
                 List of pairs of (gradient, variable) where the gradient has been averaged
                 across all towers.
            Reference: https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py
        '''
        with tf.name_scope('average_gradients'):
            average_grads = []
            for grad_and_vars in zip(*tower_grads):
                grads = []
                for g, v in grad_and_vars:
                    if g is None:
                        print 'skipped variable %s because of None gradient' % v
                        continue
                    expanded_g = tf.expand_dims(g, 0)
                    grads.append(expanded_g)
                if len(grads)<1:
                    continue
                grad = tf.concat(axis=0, values=grads)
                grad = tf.reduce_mean(grad, 0)
                v = grad_and_vars[0][1]
                grad_and_var = (grad, v)
                average_grads.append(grad_and_var)
        return average_grads
    
    # ----------------------------------------------
    @abstractmethod
    def run(self):
        ''' 
            The API for experiment setting. 
            Build a tensorflow computational graph for running the experiment.
        '''
        raise NotImplementedError
    
    # ----------------------------------------------
    def build_optimizer(self):
        ''' 
            get the optimizer object from the model 
        '''
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.model.global_step = self.global_step
        self.optimizer = self.model.get_optimizer()

    # ----------------------------------------------
    def build_checkpoint_saver(self):
        '''checkpoint saver in computational graph'''
        self._checkpoint_saver = tf.train.Saver(max_to_keep=self._max_checkpoints_to_keep) 

    # ----------------------------------------------   
    def load_latest_checkpoint(self, sess):
        ''' 
            load the latest checkpoint if it exists.
            Args: 
                sess: a tensorflow session object
        '''
        print('checkpoint dir: ', self._checkpoint_dir)
        if self._load_checkpoint and os.path.isdir(self._checkpoint_dir):
            checkpoint = tf.train.latest_checkpoint(self._checkpoint_dir)
            self._checkpoint_saver.restore(sess, checkpoint)
            print('checkpoint loaded!!!!!!!!', checkpoint)
    # ----------------------------------------------
    def save_checkpoint(self, sess,global_step):
        '''
            save a checkpoint
            Args: 
                sess: a tensorflow session object
                global_step: a tensor/numpy scalar indicating the current step number in the experiment
        '''
        self._checkpoint_saver.export_meta_graph(filename=os.path.join(self._checkpoint_dir, 'graph.meta'))
        self._checkpoint_saver.save(sess, 
                                    os.path.join(self._checkpoint_dir,'export'), 
                                    global_step=global_step, 
                                    write_meta_graph=False)


#----------------------------------------------------------------
# Experiment Setting Classes 
#----------------------------------------------------------------
class HoldOut(setting):
    '''
        Hold Out experiment setting is usually used in supervised learning tasks. 
        The datasets are splitted into a training set and a test set.
        The training set is used for training a model's parameters.
        The test set is used for testing the predictive performance of the model.
    '''
    def __init__(self):
        super(HoldOut, self).__init__()

    # ----------------------------------------------
    def model_grads(self, batch_queue, scope):  
        '''
            function for computing the gradients of the model. This function is used to be feed into the 'multi_gpu_grads' function as a parameter.
            Args:
                batch_queue: the queued batches of data
                scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
            Returns:
                grads: returns the list of gradients
        '''
        with tf.name_scope(scope) as scope:
            # load a batch of data from the queue
            x,y = batch_queue.dequeue()


            # build inference network
            logits = self.model.inference(x=x) 
            # build model loss computation
            loss = self.model.loss(logits= logits, labels=y, scope= scope) 
            grads = self.optimizer.compute_gradients(loss,var_list=self.model.trainable_vars)
            # adam optimizer need to create a set of variables
            self.optimizer.minimize(loss,var_list=self.model.trainable_vars)
            # add evaluation ops 
            self.build_metrics_graph(logits,y)

            return grads

    # ----------------------------------------------
    def build_metrics_graph(self, logits, y_):   
        '''
            build the tensorflow computational graph for the evaluation metrics.
            Args:
                logits: a tensor for the logits of the predictions
                y_: a tensor for ground-truth labels 
        '''
        with tf.name_scope('metrics'):
            for e in self.metrics:
                _,m = e.compute(logits=logits,labels=y_)

    # ----------------------------------------------
    def build_train_graph(self):
        """
          build the tensorflow computation graph for training of the model
          Training dataset will be feed to the model. 
          Returns:
            train_op: training operation tensor
            log_merged: all log tensors
        """
        with tf.name_scope('training'):
            # build dataset
            with tf.name_scope('dataset'):
                with tf.device('/cpu:0'):
                    x, y = self.dataset.train_batch(self.batch_size)

                batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([x, y], capacity=2 * (self.num_GPUs+1))

            # compute the gradients of the model using single/multiple GPUs 
            grads = self.multi_gpu_grads(batch_queue,self.model_grads)
                    
            #build checkpoint saver in computational graph
            self.build_checkpoint_saver()
            
            # build training ops (need to be built after creating the model saver)
            train_op = self.optimizer.apply_gradients(grads, global_step=self.global_step)

        return train_op

    # ----------------------------------------------
    def build_test_graph(self):
        """
          build a tensorflow computation graph for testing of the model
        """
        with tf.name_scope('testing'):
            with tf.name_scope('dataset'):
                with tf.device('/cpu:0'):
                    x, y = self.dataset.test_batch(self.batch_size)

                # prefetch the batches from CPU memory to GPU
                batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([x, y], capacity=(2* self.num_GPUs+1))
            # reuse the model parameter trained in training graph
            tf.get_variable_scope().reuse_variables()
            x,y = batch_queue.dequeue()
            
            # use the model to predict on test data
            logits = self.model.inference(x=x) 

            # build the evaluation ops
            metrics =self.build_metrics_graph(logits,y)
    #----------------------------------------------    
    def run(self):        
        ''' 
            The API function from `setting` class for running the experiment. 
            Build a tensorflow computational graph for running the experiment.
        '''
        # set up optimizer 
        self.build_optimizer()
        
        #build computational graph for training process   
        train_op = self.build_train_graph()

        # build computational graph for testing process   
        self.build_test_graph()

        # build logging summaries
        log_all = tf.summary.merge_all()
        
        # run computational graph for the experiment
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)

        # allocate only as much GPU memory based on runtime allocations: it starts out allocating very little memory, and as Sessions get run and more GPU memory is needed, we extend the GPU memory region needed by the TensorFlow process. Note that we do not release memory, since that can lead to memory fragmentation.
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            # initialize variables
            tf.global_variables_initializer().run()
            sess.run(tf.local_variables_initializer()) 

            # load latest checkpoint if exist
            print('checkpoint loaded?')
            self.load_latest_checkpoint(sess)

            # logger for results
            log_writer = tf.summary.FileWriter(self._log_dir, sess.graph)

            # queue runner for loading the dataset using multiple threads
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            
            # running training and testing steps
            for i in xrange(self._max_steps):
                # running a step without saving any log (to speed up)
                gstep, _ = sess.run([self.global_step,train_op])
                
                # running a step while save the logs
                if (i+1)%self.interval_log == 0:
                    # write summary
                    summary, gstep = sess.run([log_all, self.global_step])
                    log_writer.add_summary(summary, gstep)
                
                # save check point
                if (i+1)%self.interval_checkpoint == 0:
                    self.save_checkpoint(sess,gstep)
            log_writer.close()


#----------------------------------------------------------------
