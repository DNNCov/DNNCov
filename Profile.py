# Profile.py
# Profiles new models. Needs some updating before incorporating a new model.
# modified from DeepHunter Profile.py

'''
usage: python gen_diff.py -h
'''

from __future__ import print_function
from nnet import NNet
import argparse
import PIL.Image
import glob
from tqdm import tqdm
import pickle
from keras.datasets import mnist,cifar10
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import collections
import os, errno
from keras import backend as K
from keras.preprocessing import image

class DNNProfile():
    def __init__(self, model, exclude_layer=['input', 'flatten'],
                 only_layer=""):
        '''
        Initialize the model to be tested
        :param threshold: threshold to determine if the neuron is activated
        :param model_name: ImageNet Model name, can have ('vgg16','vgg19','resnet50')
        :param neuron_layer: Only these layers are considered for neuron coverage
        '''
        self.model = model
        self.outputs = []

        print('models loaded')

        # the layers that are considered in neuron coverage computation
        self.layer_to_compute = []
        for layer in self.model.layers:
            if all(ex not in layer.name for ex in exclude_layer):
                self.outputs.append(layer.output)
                self.layer_to_compute.append(layer.name)

        if only_layer != "":
            self.layer_to_compute = [only_layer]

        self.cov_dict = collections.OrderedDict()

        print("* target layer list:", self.layer_to_compute)


        for layer_name in self.layer_to_compute:
            for index in range(self.model.get_layer(layer_name).output_shape[-1]):
                # [mean_value_new, squared_mean_value, standard_deviation, lower_bound, upper_bound]
                self.cov_dict[(layer_name, index)] = [0.0, 0.0, 0.0, None, None]



    def count_layers(self):
        return len(self.layer_to_compute)

    def count_neurons(self):
        return len(self.cov_dict.items())

    def count_paras(self):
        return self.model.count_params()

    def update_coverage(self, input_data):

        inp = self.model.input
        functor = K.function(inp, self.outputs)
        outputs = functor([input_data[:]])
 
        for layer_idx, layer_name in enumerate(self.layer_to_compute):
            layer_outputs = outputs[layer_idx]

            # handle the layer output by each data
            # iter is the number of data
            for iter, layer_output in enumerate(layer_outputs):
                if iter % 1000 == 0:
                    print("*layer {0}, current/total iteration: {1}/{2}".format(layer_idx, iter + 1, len(layer_outputs)))

                for neuron_idx in range(layer_output.shape[-1]):
                    neuron_output = np.mean(layer_output[..., neuron_idx])
                    profile_data_list = self.cov_dict[(layer_name, neuron_idx)]

                    mean_value = profile_data_list[0]
                    squared_mean_value = profile_data_list[1]

                    lower_bound = profile_data_list[3]
                    upper_bound = profile_data_list[4]

                    total_mean_value = mean_value * iter
                    total_squared_mean_value = squared_mean_value * iter

                    mean_value_new = (neuron_output + total_mean_value) / (iter + 1)
                    squared_mean_value = (neuron_output * neuron_output + total_squared_mean_value) / (iter + 1)


                    standard_deviation = np.math.sqrt(abs(squared_mean_value - mean_value_new * mean_value_new))

                    if (lower_bound is None) and (upper_bound is None):
                        lower_bound = neuron_output
                        upper_bound = neuron_output
                    else:
                        if neuron_output < lower_bound:
                            lower_bound = neuron_output

                        if neuron_output > upper_bound:
                            upper_bound = neuron_output

                    profile_data_list[0] = mean_value_new
                    profile_data_list[1] = squared_mean_value
                    profile_data_list[2] = standard_deviation
                    profile_data_list[3] = lower_bound
                    profile_data_list[4] = upper_bound

                    self.cov_dict[(layer_name, neuron_idx)] = profile_data_list



    def dump(self, output_file):

        print("*profiling neuron size:", len(self.cov_dict.items()))
        for item in self.cov_dict.items():
            print(item)
        pickle_out = open(output_file, "wb")
        pickle.dump(self.cov_dict, pickle_out)
        pickle_out.close()

        print("write out profiling coverage results to ", output_file)
        print("done.")


def preprocessing_test_batch(x_test):

    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_test = x_test.astype('float32')
    x_test /= 255
    return x_test

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


# function taken from KJ-TaxiNet_vis_proof.ipynb
def downsampleImage(img):
    """
    Function for downsampling images of taxiway from 200x360x3 into 8x16x1
    """

    stride = 16  # Size of square of pixels downsampled to one grayscale value
    numPix = 16  # During downsampling, average the numPix brightest pixels in each square
    width = 256 // stride  # Width of downsampled grayscale image
    height = 128 // stride  # Height of downsampled grayscale image

    img = np.array(img)

    # Remove yellow/orange lines
    mask = ((img[:, :, 0].astype('float') - img[:, :, 2].astype('float')) > 60) & (
                (img[:, :, 1].astype('float') - img[:, :, 2].astype('float')) > 30)
    img[mask] = 0

    # Convert to grayscale, crop out nose, sky, bottom of image, resize to 256x128, scale so
    # values range between 0 and 1
    img = np.array(PIL.Image.fromarray(img).convert('L').crop((55, 5, 360, 140)).resize((256, 128))) / 255.0

    # Downsample image
    # Split image into stride x stride boxes, average numPix brightest pixels in that box
    # As a result, img2 has one value for every box
    img2 = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            img2[i, j] = np.mean(
                np.sort(img[stride * i:stride * (i + 1), stride * j:stride * (j + 1)].reshape(-1))[-numPix:])

    # Ensure that the mean of the image is 0.5 and that values range between 0 and 1
    # The training data only contains images from sunny, 9am conditions.
    # Biasing the image helps the network generalize to different lighting conditions (cloudy, noon, etc)
    img2 -= img2.mean()
    img2 += 0.5
    img2[img2 > 1] = 1
    img2[img2 < 0] = 0
    return img2


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

if __name__ == '__main__':

    # argument parsing
    parser = argparse.ArgumentParser(description='neuron output profiling')
    parser.add_argument('-model', help="target model to profile")
    
    parser.add_argument("--mnist", dest="mnist", help="MNIST dataset", action="store_true")
    parser.add_argument("--normalized-input", dest="normalized", help="To normalize the input", action="store_true")
    parser.add_argument("--cifar", dest="cifar10", help="CIFAR-10 dataset", action="store_true")
    parser.add_argument("--tinytaxinet", dest="tinytaxinet", help="TinyTaxiNet dataset", action="store_true")
    parser.add_argument('-output_path', help="output path")
    
    #######################################################################################################################
    parser.add_argument("--inputs", dest="inputs", default=None,
                    help="the test inputs directory", metavar="DIR")
    parser.add_argument("--input-rows", dest="img_rows", default="224",
                        help="input rows", metavar="INT")
    parser.add_argument("--input-cols", dest="img_cols", default="224",
                        help="input cols", metavar="INT")
    parser.add_argument("--input-channels", dest="img_channels", default="3",
                        help="input channels", metavar="INT")
    #######################################################################################################################
    args = parser.parse_args()
    img_rows, img_cols, img_channels = int(args.img_rows), int(args.img_cols), int(args.img_channels)

    # if statement may need to be updated briefly when incorporating new models
    if args.tinytaxinet:
        nnet = NNet("KJ_TaxiNet.nnet")
        model = Sequential(name = 'KJ_TaxiNet')
        #model.
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
    print('Successfully loaded', model.name)
    model.summary()


    def lastWord(string):

        # split by space and converting
        # string to list and
        lis = list(string.split("/"))

        # length of list
        length = len(lis)

        # returning last element in list
        return lis[length - 1]


    # Driver code
    print(lastWord(args.output_path))
    make_sure_path_exists(args.output_path)
    profiling_dict_result ="{0}/{1}.pickle".format(args.output_path,lastWord(args.output_path))
    print("profiling output file name {0}".format(profiling_dict_result))


    # get the training data for profiling

    # if statement may also need updating when incorporating new models/datasets
    if args.mnist:
        (x_train, train_label), (x_test, test_label) = mnist.load_data()
        x_train = mnist_preprocessing(x_train)
    elif args.cifar10:
        (x_train, train_label), (x_test, test_label) = cifar10.load_data()
        x_train = cifar_preprocessing(x_train)
    elif args.tinytaxinet:
        eval_folder = 'taxinetdataset/data-train/'

        # Use each example image in folder
        exampleImages = glob.glob(eval_folder + "*png")
        # print("Glob:\n")
        # print(exampleImages)
        imgNums = sorted([int(f.split("\\")[-1].split(".")[0]) for f in exampleImages])
        # print(imgNums)

        x = list()
        # counter = 0
        for imgNum in tqdm(imgNums):
            img = PIL.Image.open("{}{}.png".format(eval_folder, imgNum))
            dsImg = downsampleImage(img)
            img.close()
            x.append(dsImg)
            # if counter % 1000 == 0:
            #     print(dsImg)
            # counter += 1
        x_train = np.array(x)
        x_train = x_train.reshape(x_train.shape[0], 128)
        print(x_train.shape)
    ################################TODO: EXTENSION########################################################
    elif not args.inputs==None: 
        eval_folder = args.inputs
        # Use each example image in folder
        exampleImages = glob.glob(eval_folder + "*png")

        xs = list()
        for fname in exampleImages:
            if fname.endswith('.jpg') or fname.endswith('.png') or fname.endswith('.JPEG'):
                if img_channels==1:
                  x=image.load_img(fname, target_size=(img_rows, img_cols), color_mode = "grayscale")
                  x=np.expand_dims(x,axis=2)
                else:
                  x=image.load_img(fname, target_size=(img_rows, img_cols))
            x=np.expand_dims(x,axis=0)
            xs.append(x)
        xs=np.vstack(xs)
        x_train = xs.reshape(xs.shape[0], img_rows, img_cols, img_channels)
    else:
        print('Please extend the new train data here!')
############################################################################################################


    profiler = DNNProfile(model)

    print(np.shape(x_train))

    profiler.update_coverage(x_train)

    profiler.dump(profiling_dict_result)
