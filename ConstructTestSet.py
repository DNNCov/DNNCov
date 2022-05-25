# ConstructTestSet.py
# Constructs the test set for the model given the model type and the desired path for the output directory.
# modified from DeepHunter ConstructInitialSeeds.py

#!/usr/bin/env python2.7
import argparse
import os
import sys
from keras.datasets import mnist
from tqdm import tqdm
import PIL.Image
from Profile import downsampleImage
import glob
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing import image

sys.path.append('../')


def createBatch(x_batch, output_path, prefix):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    batches = np.split(x_batch, len(x_batch), axis=0)
    for i, batch in enumerate(batches):
        test = np.append(batch, batch, axis=0)
        saved_name = prefix + str(i) + '.npy'
        np.save(os.path.join(output_path, saved_name), test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='control experiment')
    parser.add_argument("--mnist", dest="mnist", help="MNIST dataset", action="store_true")
    parser.add_argument("--normalized-input", dest="normalized", help="To normalize the input", action="store_true")
    parser.add_argument("--cifar", dest="cifar10", help="CIFAR-10 dataset", action="store_true")
    parser.add_argument("--tinytaxinet", dest="tinytaxinet", help="TinyTaxiNet dataset", action="store_true")
    parser.add_argument('-output_path', help='Out path')
    #######################################################################################################################
    parser.add_argument("--inputs", dest="inputs", default=None,
                    help="the test inputs directory", metavar="DIR")
    parser.add_argument("--input-rows", dest="img_rows", default="224",
                        help="input rows", metavar="INT")
    parser.add_argument("--input-cols", dest="img_cols", default="224",
                        help="input cols", metavar="INT")
    parser.add_argument("--input-channels", dest="img_channels", default="3",
                        help="input channels", metavar="INT")
    ####################################################################################################################
    args = parser.parse_args()
    img_rows, img_cols, img_channels = int(args.img_rows), int(args.img_cols), int(args.img_channels)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if args.mnist:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    elif args.cifar10:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif args.tinytaxinet:
        eval_folder = './taxinetdataset/data-val/'
        # Use each example image in folder
        exampleImages = glob.glob(eval_folder + "*png")
        imgNums = sorted([int(f.split("\\")[-1].split(".")[0]) for f in exampleImages])

        x = list()
        for imgNum in tqdm(imgNums):
            img = PIL.Image.open("{}{}.png".format(eval_folder, imgNum))
            dsImg = downsampleImage(img)
            img.close()
            x.append(dsImg)
        x_test = np.array(x)
        x_test = x_test.reshape(x_test.shape[0], 128)
        print(x_test.shape)  
    ##############################TODO: EXTENSION WORK ################################################
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
        x_test = xs.reshape(xs.shape[0], img_rows, img_cols, img_channels)
    #######################################################################################################
    createBatch(x_test, args.output_path, str(0)+'_')

    print('finished creating test set')
