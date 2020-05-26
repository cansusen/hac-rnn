from expConfig import expConfig
from setting import *
from model import *
from dataset import *
from metric import * 
import argparse

#-----------------------------------------------------------------------------------------------
# parse the parameters to run from terminal arguments
#-----------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--taskid', type=int, default=0, help='the experiment task to run')
parser.add_argument('--num_gpu', type=int, default=2, help='the number of GPUs to use')
args = parser.parse_args()

# parse parameters
t = args.taskid
g = args.num_gpu

#-------------------------------------------------------------------
# classification models on text for clinical notes  
#-------------------------------------------------------------------
if t==41:
    d = ClinicalTS_numeric_combined() 
    s = HoldOut()
    s.num_GPUs = g
    e = [Accuracy()]
    m = RecurrentNN()
    p = expConfig(dataset=d,
                  setting=s,
                  model=m,
                  metrics=e)
    p.run()  

if t==42:
    d = ClinicalTS_numeric_combined() 
    s = HoldOut()
    s.num_GPUs = g
    e = [Accuracy()]
    m = FullyConnected()
    p = expConfig(dataset=d,
                  setting=s,
                  model=m,
                  metrics=e)
    p.run()  

#-------------------------------------------------------------------
if t==43:
    d = ClinicalTS_numeric_sequential() 
    s = HoldOut()
    s.num_GPUs = g
    e = [Accuracy()]
    m = RecurrentNN_wAtt()
    p = expConfig(dataset=d,
                  setting=s,
                  model=m,
                  metrics=e)
    p.run()  


if t==44:
    d = ClinicalTS_numeric_sequential() 
    s = HoldOut()
    s.num_GPUs = g
    e = [Accuracy()]
    m = RecurrentNN()
    p = expConfig(dataset=d,
                  setting=s,
                  model=m,
                  metrics=e)
    p.run()  
    
#-------------------------------------------------------------------
# Reserved for Clinical Notes Additional Run
#-------------------------------------------------------------------
if t==45:
    d = ClinicalNotes()
    s = HoldOut()
    s.num_GPUs = g
    e = [Accuracy()]
    m = SoftmaxRegression()
    p = expConfig(dataset=d,
                  setting=s,
                  model=m,
                  metrics=e)
    p.run()

if t==46:
    d = ClinicalNotes()
    s = HoldOut()
    s.num_GPUs = g
    e = [Accuracy()]
    m = FullyConnected()
    p = expConfig(dataset=d,
                  setting=s,
                  model=m,
                  metrics=e)
    p.run()

if t==47:
    d = ClinicalNotes()
    s = HoldOut()
    s.num_GPUs = g
    e = [Accuracy()]
    m = RecurrentNN()
    p = expConfig(dataset=d,
                  setting=s,
                  model=m,
                  metrics=e)
    p.run()  
 
#-------------------------------------------------------------------
# ClinicalTS_combined()
#-------------------------------------------------------------------
if t==48:
    d = ClinicalTS_combined()
    s = HoldOut()
    s.num_GPUs = g
    e = [Accuracy()]
    m = SoftmaxRegression()
    p = expConfig(dataset=d,
                  setting=s,
                  model=m,
                  metrics=e)
    p.run()  

if t==49:
    d = ClinicalTS_combined()
    s = HoldOut()
    s.num_GPUs = g
    e = [Accuracy()]
    m = FullyConnected()
    p = expConfig(dataset=d,
                  setting=s,
                  model=m,
                  metrics=e)
    p.run()  
    
if t==50:
    d = ClinicalTS_combined()
    s = HoldOut()
    s.num_GPUs = g
    e = [Accuracy()]
    m = RecurrentNN()
    p = expConfig(dataset=d,
                  setting=s,
                  model=m,
                  metrics=e)
    p.run()  
    
if t==51:
    d = ClinicalTS_combined()
    s = HoldOut()
    s.num_GPUs = g
    e = [Accuracy()]
    m = RecurrentNN_wAtt()
    p = expConfig(dataset=d,
                  setting=s,
                  model=m,
                  metrics=e)
    p.run()  

### Use this for additional run with Combined notes    
if t==52:
    d = ClinicalTS_bow_sequential()
    s = HoldOut()
    s.num_GPUs = g
    e = [Accuracy()]
    m = RecurrentNN() 
    p = expConfig(dataset=d,
                  setting=s,
                  model=m,
                  metrics=e)
    p.run()  

#-------------------------------------------------------------------
# ClinicalTS_individual()
#-------------------------------------------------------------------
if t==53:
    d = ClinicalTS_individual()
    s = HoldOut()
    s.num_GPUs = g
    e = [Accuracy()]
    m = SoftmaxRegression()
    p = expConfig(dataset=d,
                  setting=s,
                  model=m,
                  metrics=e)
    p.run()  

if t==54:
    d = ClinicalTS_individual()
    s = HoldOut()
    s.num_GPUs = g
    e = [Accuracy()]
    m = FullyConnected()
    p = expConfig(dataset=d,
                  setting=s,
                  model=m,
                  metrics=e)
    p.run()  
    
if t==55:
    d = ClinicalTS_individual()
    s = HoldOut()
    s.num_GPUs = g
    e = [Accuracy()]
    m = RecurrentNN()
    p = expConfig(dataset=d,
                  setting=s,
                  model=m,
                  metrics=e)
    p.run()  
    
if t==56:
    d = ClinicalTS_individual()
    s = HoldOut()
    s.num_GPUs = g
    e = [Accuracy()]
    m = RecurrentNN_wAtt()
    p = expConfig(dataset=d,
                  setting=s,
                  model=m,
                  metrics=e)
    p.run()  

#---------------------------------
# Hierarchical
#---------------------------------
if t==57:
    d = ClinicalTS_individual_7()
    s = HoldOut()
    s.num_GPUs = g
    e = [Accuracy()]
    m = Hierarchical_RNN()
    p = expConfig(dataset=d,
                  setting=s,
                  model=m,
                  metrics=e)
    p.run() 

if t==58:
    d = ClinicalTS_individual_8()
    s = HoldOut()
    s.num_GPUs = g
    e = [Accuracy()]
    m = Hierarchical_RNN()
    p = expConfig(dataset=d,
                  setting=s,
                  model=m,
                  metrics=e)
    p.run() 

#---------------------------------
# Hierarchical with Attention
#---------------------------------
if t==64:
    d = ClinicalTS_individual_1() 
    s = HoldOut()
    s.num_GPUs = g
    e = [Accuracy()]
    m = Hierarchical_RNN_wAtt() 
    p = expConfig(dataset=d,
                  setting=s,
                  model=m,
                  metrics=e)
    p.run()
if t==65:
    d = ClinicalTS_individual_4() 
    s = HoldOut()
    s.num_GPUs = g
    e = [Accuracy()]
    m = Hierarchical_RNN_wAtt() 
    p = expConfig(dataset=d,
                  setting=s,
                  model=m,
                  metrics=e)
    p.run()

if t==66:
    d = ClinicalTS_individual_7() 
    s = HoldOut()
    s.num_GPUs = g
    e = [Accuracy()]
    m = Hierarchical_RNN_wAtt() 
    p = expConfig(dataset=d,
                  setting=s,
                  model=m,
                  metrics=e)
    p.run()

#-------
if t==67:
    d = ClinicalTS_individual_1() 
    s = HoldOut()
    s.num_GPUs = g
    e = [Accuracy()]
    m = Hierarchical_RNN_wDoubleAtt()
    p = expConfig(dataset=d,
                  setting=s,
                  model=m,
                  metrics=e)
    p.run() 
if t==68:
    d = ClinicalTS_individual_4() 
    s = HoldOut()
    s.num_GPUs = g
    e = [Accuracy()]
    m = Hierarchical_RNN_wDoubleAtt()
    p = expConfig(dataset=d,
                  setting=s,
                  model=m,
                  metrics=e)
    p.run() 

