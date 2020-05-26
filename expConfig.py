import tensorflow as tf
import os

class expConfig:
    '''
        expConfig is used to configure an experiment, 
        setting up the elements: dataset, model, experiment setting, and evaluation metrics.
    '''
    def __init__(self, 
                 dataset, # the dataset 
                 setting, # experiment setting to run the experiment
                 model, # the model to be tested 
                 metrics=None, # the evaluation metrics 
                 skip_if_log_exist = False # whether or not to skip the experiment is the result of this experiment has already exist. If False, the new result will overwrite the old.
                ):
        self.dataset = dataset
        self.setting = setting
        self.model = model
        self.metrics = metrics
        self.skip_if_log_exist = skip_if_log_exist
        
    def run(self):
        '''
            run the experiment
        '''
        log_dir = self.log_dir
        checkpoint_dir = self.checkpoint_dir
        
        # create folder for saving results (log)
        if os.path.isdir(log_dir):
            if self.skip_if_log_exist:
                print "log exists, experiment skipped"
                return
        else:
            tf.gfile.MakeDirs(log_dir)
         
        # set up the experiment setting
        self.setting.setup(dataset=self.dataset,
                           model=self.model,
                           metrics=self.metrics,
                           log_dir=log_dir,
                           checkpoint_dir = checkpoint_dir
                          )

        # run experiment setting
        self.setting.run()
    # -----------------------------------------------
    @property
    def log_dir(self):
        return os.path.join('log',
                            self.setting_dir, 
                            self.dataset_dir,
                            self.model_dir)

    @property
    def checkpoint_dir(self):
        return os.path.join('checkpoints',
                            self.setting_dir, 
                            self.dataset_dir,
                            self.model_dir)
    
    @property
    def setting_dir(self):
        return self.attr_to_str(self.setting, 'SETTING', ['name','desc','model','dataset','metrics'])

    @property
    def dataset_dir(self):
        return self.attr_to_str(self.dataset, 'DATASET')

    @property
    def model_dir(self):
        return self.attr_to_str(self.model, 'MODEL')

    # -----------------------------------------------
    def attr_to_str(self, obj, prefix, exclude_list=['name','desc']):
        '''
            return convert the attributes of an object into a unique string of path for result log and model checkpoint saving.
            The private attributes (starting with '_', e.g., '_attr') and the attributes in the `exclude_list` will be excluded from the string.
            Args:
                obj: the object to extract the attribute values
                prefix: the prefix of the string (e.g., MODEL, DATASET) 
                exclude_list: the list of attributes to be exclud/ignored in the string 
            Returns: a unique string of path with the attribute-value pairs of the input object
        '''
        out_dir = prefix+"-"+obj.name
        for k, v in obj.__dict__.items():
            if not k.startswith('_') and k not in exclude_list:
                out_dir += "/%s-%s" % (k, ",".join([str(i) for i in v]) if type(v) == list else v)
        return out_dir

