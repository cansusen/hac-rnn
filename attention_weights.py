import tensorflow as tf
import numpy as np

attention_weights = []

in_path = '/home/csen/work/expdeep/exp/log/SETTING-HoldOut/DATASET-yelp/disease-yelp/MODEL-RecurrentNN_wAtt/fc_activation-tanh/rnn_layers-8/rnn_activation-tanh/fc_dropout_rate-0.2/fc_layers-/rnn_cell_type-lstm/rnn_dropout_rates-0.2,0.2,0.2/fc_batchnorm-False/events.out.tfevents.1543430009.compute-0-37'
out_path = 'datasets/yelp/' 

for event in tf.train.summary_iterator(in_path):
    for value in event.summary.value:
        if  (value.tag.startswith('testing/RecurrentNN_wAtt/alphas')):
            #print(value.tag)
            if  value.HasField('simple_value'):
                #print(value.simple_value)
                attention_weights.append(value.simple_value)
        elif (value.tag.startswith('testing/metrics/accuracy/accuracy_1')):
            print('accuracy:', value.simple_value)


len(attention_weights)

#np.save(out_path + 'weights_5.npy',attention_weights)
