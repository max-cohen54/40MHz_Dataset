import numpy as np
import matplotlib.pyplot as plt
import math
import h5py
import tensorflow as tf

def load_model(model_name, custom_objects=None):
    name = model_name + '.json'
    json_file = open(name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json, custom_objects=custom_objects)
    model.load_weights(model_name + '.h5')
    return model

def save_model(model_save_name, model):
    with open(model_save_name + '.json', 'w') as json_file:
        json_file.write(model.to_json())
    model.save_weights(model_save_name + '.h5')

def find_min_max_range(true, pred):
    minRange = min(true)
    minPred = min(pred)
    if minPred < minRange: minRange = minPred
        
    maxRange = max(true)
    maxPred = max(pred)
    if maxPred > maxRange: maxRange = maxPred
        
    return (minRange, maxRange)

def make_feature_plots(true, prediction, xlabel, particle, bins, density, ranges=None):
    if ranges == None: ranges = find_min_max_range(true, prediction)
        
    plt.figure(figsize=(7,5))
    plt.hist(prediction, bins=bins, histtype='step', density=density, range = ranges)
    plt.hist(true, bins=bins, histtype='step', density=density, range = ranges)
    plt.yscale('log', nonpositive='clip')
    plt.ylabel('Prob. Density(a.u.)')
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.legend([particle+' Predicted', particle+' True'])
    plt.show()
    
def mse_loss(true, prediction):
    loss = tf.reduce_mean(tf.math.square(true - prediction),axis=-1)
    return loss
