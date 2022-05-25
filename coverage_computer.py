# coverage_computer.py
# The runner file for producing the coverage report.
# Derived from DeepHunter image_fuzzer.py code

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import sys

import matplotlib.pyplot as plt

from nnet import NNet
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Input, InputLayer
from keras.layers import Activation, Flatten, BatchNormalization

from Profile import downsampleImage

import argparse, pickle
import shutil

import csv

from keras.models import load_model
import tensorflow as tf
import os
sys.path.append('../')
from keras import Input
from coverage import Coverage
from keras import backend as K
from keras.applications import mobilenet, vgg19, resnet

from keras.applications.vgg16 import preprocess_input
import random
import time
import numpy as np
from test_queue import ImageInputCorpus
from output_fetcher import build_fetch_function
from Queue import Seed
from tqdm import tqdm

from keras.models import load_model


def imagenet_preprocessing(input_img_data):
    temp = np.copy(input_img_data)
    temp = np.float32(temp)
    qq = preprocess_input(temp)
    return qq


def mnist_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.reshape(temp.shape[0], 28, 28, 1)
    temp = temp.astype('float32')
    temp /= 255
    return temp


def cifar_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        temp[:, :, :, i] = (temp[:, :, :, i] - mean[i]) / std[i]
    return temp


def tiny_taxinet_preprocessing(x_test):
    return x_test





preprocess_dic = {
    'cifar': cifar_preprocessing,
    'mnist': mnist_preprocessing,
    'tinytaxinet': tiny_taxinet_preprocessing
}


shape_dic = {
    'vgg16': (32, 32, 3),
    'resnet20': (32, 32, 3),
    'lenet1': (28, 28, 1),
    'lenet4': (28, 28, 1),
    'lenet5': (28, 28, 1),
    'mobilenet': (224, 224, 3),
    'vgg19': (224, 224, 3),
    'resnet50': (224, 224, 3),
    'tinytaxinet': (128)
}


metrics_para = {
    'kmnc': 1000,
    'bknc': 10,
    'tknc': 10,
    'nbc': 10,
    'newnc': 10,
    'nc': 0.75,
    'fann': 1.0,
    'snac': 10
}


exclude_layer_dic = {
    'vgg16.h5': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'resnet20.h5': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'lenet1.h5': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'lenet4.h5': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'lenet5.h5': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'mobilenet.h5': ['input', 'flatten', 'padding', 'activation', 'batch', 'dropout',
                  'bn', 'reshape', 'relu', 'pool', 'concat', 'softmax', 'fc'],
    'vgg19.h5': ['input', 'flatten', 'padding', 'activation', 'batch', 'dropout', 'bn',
              'reshape', 'relu', 'pool', 'concat', 'softmax', 'fc'],
    'resnet50.h5': ['input', 'flatten', 'padding', 'activation', 'batch', 'dropout', 'bn',
                 'reshape', 'relu', 'pool', 'concat', 'add', 'res4', 'res5'],
    'tinytaxinet.h5': ['input', 'flatten']
}


def metadata_function(meta_batches):
    return meta_batches


# function to retrieve the outputs given the model and the directory containing input files
# function creation in progress (see comments below)
# initially created to help MCDC implementation; unused as of now
def get_outputs(indir, model, exclude_layer):
    test_list = os.listdir(indir)

    for test_name in tqdm(test_list):
        path = os.path.join(indir, test_name)
        img = np.load(path)
        img_batches = img[1:2]

        # _, img_batches, _, _, _ = input_batches
        # if len(img_batches) == 0:
        #     return None, None
        preprocessed = preprocess(img_batches)

        inp = model.input
        outputs = []

        for layer in model.layers:
            if all(ex not in layer.name for ex in exclude_layer):
                outputs.append(layer.output)
        outputs.append(model.layers[-1].output)

        functor = K.function([inp] + [K.learning_phase()], outputs)
        layer_outputs = functor([preprocessed, 0])

        prediction_result = np.expand_dims(np.argmax(layer_outputs[-1], axis=1),axis=0)

        # coverage_batches is the output of internal layers and metadata_batches is the output of the prediction result
        # coverage_batches, metadata_batches = fetch_function((0, input_batches, 0, 0, 0))

        # TODO: create array for the outputs of each layer and return outside of loop
        # TODO: also may need to iterate through outputs as done in coverage.py to extract exact outputs
        return layer_outputs, prediction_result


# function that computes the final coverages
def dry_run(indir, fetch_function, coverage_function, queue, f, nccsv):
    test_list = os.listdir(indir)
    # Read each initial seed and analyze the coverage
    counter = 1

    # arrays to hold the coverage percentages over time (in order: kMNC, TKNC, NBC, SNAC, NC)
    # used in plotting the coverages over the iterations
    cov0 = [0]
    cov1 = [0]
    cov2 = [0]
    cov3 = [0]
    cov4 = [0]

    print("\nTotal number of tests in test set: " + str(len(test_list)))
    f.write("\nTotal number of tests in test set: " + str(len(test_list)))
    nccsv.write("\n" + str(len(test_list)) + "\n")
    print("\nCOVERAGE REPORT:")
    f.write("\n\nCOVERAGES:")
    for test_name in tqdm(test_list):
        if counter % 1000 == 0 and counter < len(test_list):
            print("\nCurrent coverages (~" + str(counter) + " test images): [KMNC %, TKNC %, NBC %, SNAC %, NC %] = " +
                  queue.dry_run_cov_str)
            f.write("\nCurrent coverages (~" + str(counter) + " test images): [KMNC %, TKNC %, NBC %, SNAC %, NC %] = " +
                  queue.dry_run_cov_str)
        if counter > 1:
            # updating the coverage over time arrays
            cov0.append(queue.dry_run_cov[0])
            cov1.append(queue.dry_run_cov[1])
            cov2.append(queue.dry_run_cov[2])
            cov3.append(queue.dry_run_cov[3])
            cov4.append(queue.dry_run_cov[4])
        path = os.path.join(indir, test_name)
        img = np.load(path)
        # Each seed will contain two images, i.e., the reference image and mutant (see the paper)
        input_batches = img[1:2]
        # Predict the mutant and obtain the outputs
        # coverage_batches is the output of internal layers and metadata_batches is the output of the prediction result
        coverage_batches, metadata_batches = fetch_function((0, input_batches, 0, 0, 0))
        # Based on the output, compute the coverage information
        coverage_list = coverage_function(coverage_batches)
        metadata_list = metadata_function(metadata_batches)
        # Create a new seed
        input = Seed(0, coverage_list, test_name, None, metadata_list[0][0], metadata_list[0][0])
        new_img = np.append(input_batches, input_batches, axis=0)
        # Put the seed in the queue and save the npy file in the queue dir
        # save_if_interesting is the function that performs the coverage updating given the new seed
        queue.save_if_interesting(input, new_img, False, True, test_name)
        counter += 1

    # Below is example code to plot coverage over iterations (time); cov4 represents NC, as indicated in the order
    #   above. The code can be copied and pasted to plot all coverage types, but it is currently commented out in order
    #   to allow quicker computations during the testing process.
    # plt.plot(cov4)
    # plt.ylabel('Coverage')
    # plt.show()

    # need to put criteria for nbc and snac? what do they mean?
    print("\nFINAL COVERAGES:")
    print("k-Multisection Neuron Coverage (k: " + str(cri[0]) + ") = " + str(queue.dry_run_cov[0]) + "%")
    print("Top-k Neuron Coverage (k: " + str(cri[1]) + ") = " + str(queue.dry_run_cov[1]) + "%")
    print("Neuron Boundary Coverage = " + str(queue.dry_run_cov[2]) + "%")
    print("Strong Neuron Activation Coverage = " + str(queue.dry_run_cov[3]) + "%")
    print("Neuron Coverage (threshold: " + str(cri[4]) + ") = " + str(queue.dry_run_cov[4]) + "%")

    f.write("\n\nFINAL COVERAGES:")
    f.write("\nk-Multisection Neuron Coverage (k: " + str(cri[0]) + ") = " + str(queue.dry_run_cov[0]) + "%")
    f.write("\nTop-k Neuron Coverage (k: " + str(cri[1]) + ") = " + str(queue.dry_run_cov[1]) + "%")
    f.write("\nNeuron Boundary Coverage = " + str(queue.dry_run_cov[2]) + "%")
    f.write("\nStrong Neuron Activation Coverage = " + str(queue.dry_run_cov[3]) + "%")
    f.write("\nNeuron Coverage (threshold: " + str(cri[4]) + ") = " + str(queue.dry_run_cov[4]) + "%")


if __name__ == '__main__':
    #start_time = time.time()
    #tf.logging.set_verbosity(tf.logging.INFO)
    random.seed(time.time())

    parser = argparse.ArgumentParser(description='coverage computer for DNN')
    
    parser.add_argument('-model', help="target model to profile")
    parser.add_argument("--mnist", dest="mnist", help="MNIST dataset", action="store_true")
    parser.add_argument("--normalized-input", dest="normalized", help="To normalize the input", action="store_true")
    parser.add_argument("--cifar", dest="cifar10", help="CIFAR-10 dataset", action="store_true")
    parser.add_argument("--tinytaxinet", dest="tinytaxinet", help="TinyTaxiNet dataset", action="store_true")
    
    parser.add_argument('-output_path', help="output path")

    
    
    parser.add_argument('-i', help='input test set directory')
    #parser.add_argument('--profile', dest='profile', default='-1', help='the profile path')
    # are the ones apart from 'all' necessary? only necessary for saving time when only one criteria is needed
    #parser.add_argument('-criteria', help="set the coverage criteria",
    #                    choices=['nc', 'kmnc', 'nbc', 'snac', 'bknc', 'tknc', 'all'], default='all')
   
    # consider the possibility that user selects 'all' and desires certain metric parameters
    # TODO: format the below argument so that it accepts an array that holds the metric parameter for each criteria
    #       (currently does not function)
    #parser.add_argument('-metric_para', help="set the parameter for different metrics", type=float)

    args = parser.parse_args()

    img_rows, img_cols = 224, 224
    input_shape = (img_rows, img_cols, 3)
    input_tensor = Input(shape=input_shape)

    # Get the layers which will be excluded during the coverage computation
    exclude_layer_list = exclude_layer_dic[args.model]

    # Create the output directory including seed queue and crash dir, it is like AFL
    #if os.path.exists(args.o):
    #    shutil.rmtree(args.o)
    #os.makedirs(os.path.join(args.o, 'queue'))
    # TODO: remove crashes directory (unnecessary)
    #os.makedirs(os.path.join(args.o, 'crashes'))

    # coverage report file
    reportPath = args.output_path + '//CoverageReport'
    f = open(reportPath, "w")
    f.close()
    f = open(reportPath, "a")
    f.write("COVERAGE REPORT\n")
    f.write("_____________________________________________________________________________________________________\n\n")

    # NC CSV file for visualization purposes
    # Currently a work in progress (as explained previously, needs to be edited so that (a) models other than LeNet-5
    #   can be used and (b) CSV files can be created for criteria other than NC)
    csvPath = args.output_path + '//NC_CSV'
    nccsv = open(csvPath, "w")
    nccsv.close()
    nccsv = open(csvPath, "a")

    # Load model. For ImageNet, we use the default models from Keras framework.
    # For other models, we load the model from the h5 file.
    model = None
    if args.tinytaxinet:
        nnet = NNet('KJ_TaxiNet.nnet')
        model = Sequential(name = 'KJ_TaxiNet')
        # consider having the model below saved as h5 instead for determinism and to save time
        for ind, layer_size in enumerate(nnet.layerSizes[1:-1]):
            if ind == 0:
                model.add(Dense(layer_size, activation='relu', kernel_initializer='he_normal', input_shape=(128,),
                                name='dense_{}'.format(ind + 1)))
            else:
                model.add(Dense(layer_size, activation='relu', kernel_initializer='he_normal',
                                name='dense_{}'.format(ind + 1)))

        model.add(Dense(2, kernel_initializer='he_normal', name='dense_4'))
        model = Sequential(name='KJ_TaxiNet')
        model = load_model(args.model)
        layer_name_list = ['dense_1', 'dense_2', 'dense_3', 'dense_4']
        for ind, layer_name in enumerate(layer_name_list):
            temp = model.get_layer(layer_name).get_weights()
            l_w = np.transpose(np.array(nnet.weights[ind]))
            l_b = np.array(nnet.biases[ind])
            model.get_layer(layer_name).set_weights([l_w, l_b])

    else:
        model = load_model(args.model)

    # Get the preprocess function based on different dataset
    preprocess=None
    if args.mnist:
        preprocess = preprocess_dic['mnist']    
    if args.cifar10:
        preprocess = preprocess_dic['cifar']
    if args.tinytaxinet:
        preprocess = preprocess_dic['tinytaxinet']

    # Load the profiling information which is needed by the metrics in DeepGauge
    #print(args.model)
    #print(model_profile_path[args.model])
    profile_dict = pickle.load(open('{0}/{0}.pickle'.format(args.output_path), 'rb'))

    # Load the configuration for the selected metrics.
    # kMNC, TKNC, NBC, SNAC, NC
    cri = [0, 0, 0, 0, 0]
    
    cri[0] = metrics_para['kmnc']
    cri[1] = metrics_para['tknc']
    cri[2] = metrics_para['nbc']
    cri[3] = metrics_para['snac']
    cri[4] = metrics_para['nc']
    # The coverage computer
    coverage_handler = Coverage(model=model, criteria="all", k=cri,
                                profiling_dict=profile_dict, exclude_layer=exclude_layer_list)

    model_names = [args.model]

    # fetch_function is to perform the prediction and obtain the outputs of each layers
    dry_run_fetch = build_fetch_function(coverage_handler, preprocess)

    # The function to update coverage
    coverage_function = coverage_handler.update_coverage

    # The test set queue
    queue = ImageInputCorpus(args.output_path, coverage_handler.total_size, "all")

    f.write("Model name: " + args.model)
    if args.mnist:
        print("Dataset: " + "MNIST")
    if args.cifar10:
        print("Dataset: " + "CIFAR10")
    if args.tinytaxinet:
        print("Dataset: " + "TinyTaxiNet")
    
    print("Model: " + args.model)
    
    f.write("\n\nModel summary:\n")
    
    print("\nModel summary:")
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(model.summary())

    nccsv.write(args.model)
    nccsv.write("\nNeuron Coverage (NC)")

    # Perform the dry_run process from the test inputs
    dry_run(args.i, dry_run_fetch, coverage_function, queue, f, nccsv)

    # writes the nccsv array in CSV fashion to the desired file
    csvWriter = csv.writer(nccsv, delimiter=',')
    csvWriter.writerows(coverage_handler.nccsvArr)

    #print('\nTime taken (seconds): ', time.time() - start_time)

    f.close()
    nccsv.close()
