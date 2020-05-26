from abc import ABCMeta, abstractmethod
import tensorflow as tf

class metric(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.values=[]
    
    @abstractmethod
    def compute(self,logits, labels):
        pass
    # ----------------------------------------------
    # list of properties and their setters
    @classmethod
    def class_name(cls):
        return cls.__name__
    @property
    def name(self):
        return self.class_name()



#----------------------------------
class Accuracy(metric):
    def compute(self,logits, labels):
        """
          Evaluate the quality of the logits at predicting the label.
          Args:
            logits: Logits tensor, float - [batch_size, NUM_CLASSES].
            labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
          Returns:
            Summary of accuracy.
        """
        with tf.name_scope('accuracy'):
            labels = tf.to_int32(labels)
            correct = tf.nn.in_top_k(logits, labels, 1)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32),name='accuracy')
            acc_summary=tf.summary.scalar('accuracy', accuracy)
        return accuracy, acc_summary

#----------------------------------
#----------------------------------
class AUC(metric):
    def compute(self,logits, labels):
        """
          Evaluate the quality of the logits at predicting the label.
          Args:
            logits: Logits tensor, float - [batch_size, NUM_CLASSES].
            labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
          Returns:
            Summary of accuracy.
        """
        with tf.name_scope('auc'):
            labels      = tf.to_int32(labels)
            auc, update_op = tf.metrics.auc(labels, logits[:,1], name='auc')
            #auc         = roc_auc_score(logits[:,1], labels)
            acc_summary = tf.summary.scalar('auc', auc)
        return auc, update_op


    
    
    
    
