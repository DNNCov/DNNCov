from tensorflow.keras.datasets import cifar10
import time
import PIL
import pandas as pd
from tensorflow.keras.layers import InputLayer
import tensorflow as tf
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from itertools import combinations
import glob
from tensorflow.keras import backend as K
from nnet import NNet
from keras.preprocessing import image

### IMPORTANT PARAMETERS FOR IMAGE PROCESSING ###
stride = 16  # Size of square of pixels downsampled to one grayscale value
numPix = 16  # During downsampling, average the numPix brightest pixels in each square
width = 256 // stride  # Width of downsampled grayscale image
height = 128 // stride  # Height of downsampled grayscale image
def predict(models,x):
    model = models

    # If skippable is not specified, use default skippable layers.
    skippable = [InputLayer, Flatten]

    # Functors that returns the outputs of the not skippable layers.
    functors = Model(inputs=model.input,
                     outputs=[l.output for l in model.layers if type(l) not in skippable])
    outs = [K.mean(K.reshape(l, (-1, l.shape[-1])), axis=0) for l in functors(x)]

    # Return internal outputs and logits.
    internals = outs[:-1]
    logits = outs[-1]
    return internals, logits
class MCDC:
    def __init__(self, model, x1, x2, modelnum,trainlimit,testlimit):
        start_time = time.time()
        self.model = model
        self.model_params = []
        self.score = []
        self.model_params_train = []
        self.testsuite_x = x2[:testlimit]
        self.trainsuite_x = x1[:trainlimit]
        self.neuronpairs = []
        self.inputpairs = []
        self.inputpairs_train = []
        self.coveredsetss = set()
        self.coveredsetvs = set()
        self.coveredsetsv = set()
        self.coveredsetvv = set()
        self.totalset = set()
        self.modelnumber = modelnum
            #print("Initialization Time:  %s" % (time.time() - start_time))
        self.start_time_input_pairs = time.time()
        self.generate_input_pairs(len(self.testsuite_x))
        #print("Length of Testset:", len(self.testsuite_x))
        self.start_time_input_pairsfinal=time.time()-self.start_time_input_pairs

        self.start_time_input_pairs_train = time.time()
        self.generate_input_pairs_train(len(self.trainsuite_x))
        #print("Length of Trainset:", len(self.trainsuite_x))
        self.start_time_input_pairs_trainfinal=time.time()-self.start_time_input_pairs_train
   
        self.start_time_values_train = time.time()
        self.neuron_values_train()
        self.start_time_values_trainfinal = time.time() - self.start_time_values_train
        
        self.start_time_stats = time.time()
        self.threshold_calculator()
        self.stats()
        self.start_time_statsfinal=time.time()-self.start_time_stats
        
        self.start_time_values_test = time.time()
        self.neuron_values_test()
        self.start_time_values_testfinal = time.time() - self.start_time_values_test
     #########################################################################################################################
    def sc(self, input_x1, input_x2, layer_id,
           neuron_id):  # False if no sign change. Takes input1, input2, layer# and neuron#
        if ((self.model_params[input_x1][layer_id][neuron_id] <= 0 and self.model_params[input_x2][layer_id][neuron_id] > 0) or (self.model_params[input_x1][layer_id][neuron_id] > 0 and self.model_params[input_x2][layer_id][neuron_id] <= 0)):
            return True
        else:
            return False
#########################################################################################################################
    def nsc_alllayer_exceptspecific(self, input_x1, input_x2, layer_id,
                                    neuron_id):  # checks if neuronid is the only neuron in layerid which has a sign change
        for i in range(0, len(self.model_params[0][layer_id])):
            if (self.sc(input_x1, input_x2, layer_id, i) and i != neuron_id):
                return False
        return True
    #########################################################################################################################
    def nsc_alllayer(self, input_x1, input_x2, layer_id):  # checks if all neurons in layerid donot have a sign change
        for i in range(0, len(self.model_params[0][layer_id])):
            if (self.sc(input_x1, input_x2, layer_id, i)):
                return False
        return True
    #########################################################################################################################
    def vc(self, input_x1, input_x2, layer_id, neuron_id, threshold,
           option):  # False if no value change. Takes input1, input2, layer# and neuron#
        bigger = 0
        smaller = 0
        threshold = self.score[layer_id][neuron_id][1]
        if (self.model_params[input_x1][layer_id][neuron_id] >= self.model_params[input_x2][layer_id][neuron_id]):
            bigger = self.model_params[input_x1][layer_id][neuron_id]
            smaller = self.model_params[input_x2][layer_id][neuron_id]
        else:
            bigger = self.model_params[input_x2][layer_id][neuron_id]
            smaller = self.model_params[input_x1][layer_id][neuron_id]
        if (option == 1):
            if (abs(bigger - smaller) > threshold):
                return True
            else:
                return False
        if (option == 2):
            if (bigger / smaller > threshold):
                return True
            else:
                return False
    #########################################################################################################################
    def neuron_values_test(self):
        internals = []
        for i in range(0, len(self.testsuite_x)):
            if (self.modelnumber == 0):
                internals, logits = predict(self.model,
                    tf.identity([np.array(np.array(self.testsuite_x[i], dtype=object), dtype=object)]))
            elif (self.modelnumber == 1):
                internals, logits = predict(self.model,self.testsuite_x[i])
            temp = []
            for j in range(0, len(internals)):
                temp.append(list(internals[j].numpy()))
            self.model_params.append(temp)
    #########################################################################################################################
    def neuron_values_train(self):
        internals_train = []
        for i in range(0, len(self.trainsuite_x)):
            if (self.modelnumber == 0):
                internals_train, logits_train = predict(self.model,
                    tf.identity([np.array(np.array(self.trainsuite_x[i], dtype=object), dtype=object)]))
            elif (self.modelnumber == 1):
                internals_train, logits_train = predict(self.model,self.trainsuite_x[i])
            temp_train = []
            for j in range(0, len(internals_train)):
                temp_train.append(list(internals_train[j].numpy()))
            self.model_params_train.append(temp_train)
    #########################################################################################################################
    # This is code to generate input pairs
    def generate_input_pairs(self, limit):
        inputarr = []
        for i in range(0, limit):
            inputarr.append(i)
        self.inputpairs = list(combinations(inputarr, 2))
    #########################################################################################################################
    # This is code to generate input pairs
    def generate_input_pairs_train(self, limit):
        inputarr = []
        for i in range(0, limit):
            inputarr.append(i)
        self.inputpairs_train = list(combinations(inputarr, 2))
    #########################################################################################################################
    # This is code to generate neuron pairs
    def generate_neuron_pairs(self):
        self.neuronpairs = []
        for i in range(0, len(self.model_params[0]) - 1):
            for j in range(0, len(self.model_params[0][i])):
                for k in range(0, len(self.model_params[0][i + 1])):
                    self.neuronpairs.append(list((i, j, i + 1, k, 0)))
    #########################################################################################################################
    def threshold_calculator(self):
        count = 0
        for i in list(self.inputpairs_train)[:]:  # Traverse All Input Pairs
            for j in range(0, len(self.model_params_train[0])):  # Traversing Layers
                if (count == 0):
                    self.score.append([])
                for k in range(0, len(self.model_params_train[0][j])):  # Traversing Neurons
                    if (count == 0):
                        self.score[j].append([0, 0])
                    a = abs(self.model_params_train[i[0]][j][k] - self.model_params_train[i[1]][j][k])
                    self.score[j][k][0] = self.score[j][k][0] + a
            count = count + 1
    #########################################################################################################################
    def stats(self):
        for i in range(0, len(self.score)):
            for j in range(0, len(self.score[i])):
                self.score[i][j][1] = self.score[i][j][0] / len(self.inputpairs_train)

    def opt_coverage(self, ss, vs, sv, vv):
            #pbar=tqdm(total=len(self.inputpairs))
            start_time = time.time()
            self.generate_neuron_pairs()
            #print("Neuron Pairs Time: %s" % (time.time() - start_time))
            for i in self.neuronpairs:
                self.totalset.add((i[0], i[1], i[2], i[3], i[4]))
            counter = -1
            for i in list(self.inputpairs)[:]:  # Traverse All Input Pairs
                counter = counter + 1
                if (counter % 1000 == 0):
                    print("[# of Pairs,SS,SV,VS,VV]","["+str(counter)+",",  str(round(100*len(self.coveredsetss)/len(self.totalset),2))+"%,", str(round(100*len(self.coveredsetsv)/len(self.totalset),2))+"%,",str(round(100*len(self.coveredsetvs)/len(self.totalset),2))+"%,",str(round(100*len(self.coveredsetvv)/len(self.totalset),2))+"%]")
                    #print("")
                    #pbar.update(counter)
                nscarr = []
                scarr = []
                vcarr = []
                nvcarr = []
                #################################################################################################################
                for r in range(0, len(self.model_params[0])):  # Traversing Layers
                    tempsc = []
                    tempnsc = []
                    tempvc = []
                    tempnvc = []
                    for s in range(0, len(self.model_params[0][r])):  # Traversing Neurons
                        if (self.sc(i[0], i[1], r, s) == True):
                            tempsc.append(s)
                        else:
                            tempnsc.append(s)
                        if (self.vc(i[0], i[1], r, s, 5, 1) == True):
                            tempvc.append(s)
                        else:
                            tempnvc.append(s)
                    nscarr.append(tempnsc)
                    scarr.append(tempsc)
                    nvcarr.append(tempnvc)
                    vcarr.append(tempvc)
                ####################################################################################################################
                if (ss == True):
                    for j in range(0, len(scarr) - 1):  # Traverse through all layers
                        if (len(scarr[j]) == 1):  # Check if nsc condition holds
                            for k in range(0, len(scarr[j])):  # Traverse throough each sc element of array of layer j
                                for l in range(0, len(
                                        scarr[j + 1])):  # traverse through each sc element of array of layer j+1
                                    self.coveredsetss.add((j, scarr[j][k], j + 1, scarr[j + 1][l], 1))
                ####################################################################################################################
                if (vs == True):
                    for j in range(0, len(scarr) - 1):  # Traverse through all layers
                        if (len(scarr[j]) == 0):  # Check if nsc condition holds
                            for k in range(0, len(vcarr[j])):  # Traverse throough each vc element of array of layer j
                                for l in range(0, len(scarr[j + 1])):  # traverse through each sc element of array of layer j+1
                                    self.coveredsetvs.add((j, vcarr[j][k], j + 1, scarr[j + 1][l], 1))
                ####################################################################################################################
                if (sv == True):
                    for j in range(0, len(scarr) - 1):  # Traverse through all layers
                        if (len(scarr[j]) == 1):
                            for k in range(0, len(scarr[j])):  # Traverse throough each sc element of array of layer j
                                for l in range(0, len(
                                        vcarr[j + 1])):  # traverse through each vc element of array of layer j+1
                                    if (vcarr[j + 1][l] in nscarr[j + 1]):  # check if vc neuron of j+1 is in nsc of j+1
                                        self.coveredsetsv.add((j, scarr[j][k], j + 1, vcarr[j + 1][l], 1))
                ####################################################################################################################
                if (vv == True):
                    for j in range(0, len(scarr) - 1):  # Traverse through all layers
                        if (len(scarr[j]) == 0):
                            for k in range(0, len(vcarr[j])):  # Traverse throough each sc element of array of layer j
                                for l in range(0, len(
                                        vcarr[j + 1])):  # traverse through each vc element of array of layer j+1
                                    if (vcarr[j + 1][l] in nscarr[j + 1]):  # check if vc neuron of j+1 is in nsc of j+1
                                        self.coveredsetvv.add((j, vcarr[j][k], j + 1, vcarr[j + 1][l], 1))

            print("[# of Pairs,SS,SV,VS,VV]", "["+str(counter)+",",
                  str(round(100 * len(self.coveredsetss) / len(self.totalset), 2))+"%,",
                  str(round(100 * len(self.coveredsetsv) / len(self.totalset), 2))+"%,",
                  str(round(100 * len(self.coveredsetvs) / len(self.totalset), 2))+"%,",
                  str(round(100 * len(self.coveredsetvv) / len(self.totalset), 2))+"%]")


####################################################################################################################
def downsampleImage(img):
    img = np.array(img)
    mask = ((img[:, :, 0].astype('float') - img[:, :, 2].astype('float')) > 60) & (
            (img[:, :, 1].astype('float') - img[:, :, 2].astype('float')) > 30)
    img[mask] = 0
    img = np.array(PIL.Image.fromarray(img).convert('L').crop((55, 5, 360, 140)).resize((256, 128))) / 255.0
    img2 = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            img2[i, j] = np.mean(
                np.sort(img[stride * i:stride * (i + 1), stride * j:stride * (j + 1)].reshape(-1))[-numPix:])
    img2 -= img2.mean()
    img2 += 0.5
    img2[img2 > 1] = 1
    img2[img2 < 0] = 0
    return img2
def coverage_report_mcdc(dataset,trainlimit,testlimit, model_name=None, img_dims=[224,224,3]):
    start_time_all= time.time()
    model=None

    if (dataset=='mnist'):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
    elif (dataset=='cifar'):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
        x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
    elif (dataset=='tinytaxinet'):
        train_folder="taxinetdataset/data-train/"
        eval_folder="taxinetdataset/data-val/"
        table = pd.read_csv(train_folder + "errors.csv")
        exampleImages = glob.glob(train_folder + "*png")
        imgNums = sorted([int(f.split("\\")[-1].split(".")[0]) for f in exampleImages])
        x_train = list()
        y_train = list()
        for imgNum in imgNums:
            img = PIL.Image.open("{}{}.png".format(train_folder, imgNum))
            img_copy = np.array(img)
            truth = np.array([table.CTE[imgNum], table.HE[imgNum]])
            img.close()
            x_train.append(img_copy)
            y_train.append(truth)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        print("Shape of X:", x_train.shape)
        nnet = NNet("KJ_TaxiNet.nnet")
        model = keras.models.load_model("tinytaxinet.h5")
        layer_name_list = ['dense_1', 'dense_2', 'dense_3', 'dense_4']
        for ind, layer_name in enumerate(layer_name_list):
            temp = model.get_layer(layer_name).get_weights()
            l_w = np.transpose(np.array(nnet.weights[ind]))
            l_b = np.array(nnet.biases[ind])
            model.get_layer(layer_name).set_weights([l_w, l_b])
        train_coverage_nnet = []
        train_coverage_keras = []
        for imgNum, img in enumerate(x_train[:]):
            dsImg = downsampleImage(img)
            flat_img = dsImg.reshape(-1)
            flat_img_keras = np.expand_dims(flat_img, axis=0)
            train_coverage_nnet.append(flat_img)
            train_coverage_keras.append(flat_img_keras)
        table = pd.read_csv(eval_folder + "errors.csv")
        exampleImages = glob.glob(eval_folder + "*png")
        imgNums = sorted([int(f.split("\\")[-1].split(".")[0]) for f in exampleImages])
        x_test = list()
        y_test = list()
        test_coverage_nnet = []
        test_coverage_keras = []
        for imgNum in imgNums:
            img = PIL.Image.open("{}{}.png".format(eval_folder, imgNum))
            img_copy = np.array(img)
            truth = np.array([table.CTE[imgNum], table.HE[imgNum]])
            img.close()
            x_test.append(img_copy)
            y_test.append(truth)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        print("Shape of X:", x_test.shape)
        for imgNum, img in enumerate(x_test[:]):
            dsImg = downsampleImage(img)
            flat_img = dsImg.reshape(-1)
            flat_img_keras = np.expand_dims(flat_img, axis=0)
            test_coverage_nnet.append(flat_img)
            test_coverage_keras.append(flat_img_keras)
    ######################TODO: EXTENSION TO ANY DATSET########################################################### 
    elif type(dataset) is list:
        img_rows, img_cols, img_channels = img_dims[0], img_dims[1], img_dims[2]
        # test inputs
        eval_folder = dataset[0]
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
        # train inputs
        eval_folder = dataset[1]
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
    ########################################################################################################## 
    if not model_name==None:
        if(dataset=='tinytaxinet'):            
            mcdc_coverage = MCDC(model, train_coverage_keras, test_coverage_keras, 1, trainlimit, testlimit)
        else:
            model = keras.models.load_model(model_name)
            mcdc_coverage = MCDC(model, x_train, x_test, 0, trainlimit, testlimit)
        
        start_time_allfinal=time.time()-start_time_all
        print("Input Pair Generation Train Time:  %s seconds" % round(mcdc_coverage.start_time_input_pairsfinal,2))
        print("Input Pair Generation Test Time:  %s seconds" % round(mcdc_coverage.start_time_input_pairs_trainfinal,2))
        print("Neuron Values Train Time: %s seconds" % round(mcdc_coverage.start_time_values_trainfinal,2))
        print("Profile Value Change Time: %s seconds" % round(mcdc_coverage.start_time_statsfinal,2))
        print("Neuron Values Test Time: %s seconds" % round(mcdc_coverage.start_time_values_testfinal,2))
        print("Total Load Time: %s seconds" % round(start_time_allfinal,2))
        
        start_time=time.time()
        mcdc_coverage.opt_coverage(True, True, True, True)
        print("Dataset = " + dataset)
        print("Architecture = " + model_name)
        
        print(model.summary())
        print("SIGN-SIGN Coverage = " + str(len(mcdc_coverage.coveredsetss)) + " (" + str(
            round(100 * len(mcdc_coverage.coveredsetss) / len(mcdc_coverage.totalset),
                  2)) + "%" + ")")  # ,coveredsetss)
        print("SIGN-VALUE Coverage = " + str(len(mcdc_coverage.coveredsetsv)) + " (" + str(
            round(100 * len(mcdc_coverage.coveredsetsv) / len(mcdc_coverage.totalset),
                  2)) + "%" + ")")  # ,coveredsetss)
        print("VALUE-SIGN Coverage = " + str(len(mcdc_coverage.coveredsetvs)) + " (" + str(
            round(100 * len(mcdc_coverage.coveredsetvs) / len(mcdc_coverage.totalset),
                  2)) + "%" + ")")  # ,coveredsetss)
        print("VALUE-VALUE Coverage = " + str(len(mcdc_coverage.coveredsetvv)) + " (" + str(
            round(100 * len(mcdc_coverage.coveredsetvv) / len(mcdc_coverage.totalset),
                  2)) + "%" + ")")  # ,coveredsetss)
        print("Total Pairs =", len(mcdc_coverage.totalset))
        for i in mcdc_coverage.coveredsetss:
            print(str(i[:-1][0])+","+str(i[:-1][1])+","+str(i[:-1][2])+","+str(i[:-1][3])+",")
        print("-------------------------------------------------------------------------------------------")
        for i in mcdc_coverage.coveredsetsv:
            print(str(i[:-1][0])+","+str(i[:-1][1])+","+str(i[:-1][2])+","+str(i[:-1][3])+",")
        print("-------------------------------------------------------------------------------------------")
        for i in mcdc_coverage.coveredsetvs:
            print(str(i[:-1][0])+","+str(i[:-1][1])+","+str(i[:-1][2])+","+str(i[:-1][3])+",")
        print("-------------------------------------------------------------------------------------------")
        for i in mcdc_coverage.coveredsetvv:
            print(str(i[:-1][0])+","+str(i[:-1][1])+","+str(i[:-1][2])+","+str(i[:-1][3])+",")
        print("-------------------------------------------------------------------------------------------")
       
        print("Input Pair Generation Train Time:  %s seconds" % round(mcdc_coverage.start_time_input_pairsfinal,2))
        print("Input Pair Generation Test Time:  %s seconds" % round(mcdc_coverage.start_time_input_pairs_trainfinal,2))
        print("Neuron Values Train Time: %s seconds" % round(mcdc_coverage.start_time_values_trainfinal,2))
        print("Profile Value Change Time: %s seconds" % round(mcdc_coverage.start_time_statsfinal,2))
        print("Neuron Values Test Time: %s seconds" % round(mcdc_coverage.start_time_values_testfinal,2))
        print("Total Load Time: %s seconds" % round(start_time_allfinal,2))
        print("Total Compute Time: %s seconds" % round(time.time() - start_time,2))