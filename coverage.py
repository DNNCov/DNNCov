# coverage.py
# Updates the kMNC, TKNC, NBC, SNAC, NC coverage criteria.
# modified DeepHunter coverage.py code

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import collections
from keras import backend as K
from collections import OrderedDict
import sys
class Coverage():

    def __init__(self, model, criteria, k=[1000, 10, 10, 10, 0.75], profiling_dict={},
                 exclude_layer=['input', 'flatten']):

        self.model = model

        # criteria parameters ([kMNC, TKNC, NBC, SNAC, NC])
        self.k = [0, 0, 0, 0, 0]
        self.bytearray_len = [0, 0, 0, 0, 0]

        # kmnc
        self.k[0] = k[0]
        self.bytearray_len[0] = self.k[0]

        # tknc
        self.k[1] = k[1]
        self.bytearray_len[1] = self.k[1]

        # nbc
        self.k[2] = k[2] + 1
        self.bytearray_len[2] = self.k[2] * 2

        # snac
        self.k[3] = k[3] + 1
        self.bytearray_len[3] = self.k[3]

        # nc
        self.k[4] = k[4]
        self.bytearray_len[4] = 1

        self.criteria = criteria
        self.profiling_dict = profiling_dict

        self.layer_to_compute = []
        self.outputs = []
        self.layer_neuron_num = []
        self.layer_start_index = [[], [], [], [], []]
        self.start = 0

        num = [0, 0, 0, 0, 0]
        for layer in self.model.layers:
            if all(ex not in layer.name for ex in exclude_layer):
                for i in range(5):
                    self.layer_start_index[i].append(num[i])
                self.layer_to_compute.append(layer.name)
                self.outputs.append(layer.output)
                self.layer_neuron_num.append(layer.output.shape[-1])
                for i in range(5):
                    num[i] += int(layer.output.shape[-1] * self.bytearray_len[i])
        self.outputs.append(self.model.layers[-1].output)

        # array representing the denominator for the different coverage criteria
        self.total_size = num

        # print(self.total_size)
        # print(self.layer_neuron_num)
        # print(self.layer_start_index)
        # print(self.layer_to_compute)F

        # currently creates the CSV array for LeNet-5 architecture
        # needs to be updated so that the array is created automatically for any architecture (i.e., any list of
        #   numbers representing the number of neurons in the non-excluded layers
        #self.nccsvArr = [[0]*6, [0]*6, [0]*16, [0]*16, [0]*120, [0]*84, [0]*10]
        #self.nccsvArr = [[0]*16, [0]*8, [0]*8, [0]*2, [0]*120, [0]*84, [0]*10]
        self.nccsvArr = [
        [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200,
        [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200,
        [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200,
        [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200,
        [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200,
        [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200,
        [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200,
        [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200,
        [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200,
        [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200,
        [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200,
        [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200,
        [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200,
        [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200,
        [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200,
        [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200, [0] * 200
                         ]

        self.cov_dict = collections.OrderedDict()

        inp = self.model.input
        self.functor = K.function([inp], self.outputs)


    def scale(self, layer_outputs, rmax=1, rmin=0):
        '''
        scale the intermediate layer's output between 0 and 1
        :param layer_outputs: the layer's output tensor
        :param rmax: the upper bound of scale
        :param rmin: the lower bound of scale
        :return:
        '''
        divider = (layer_outputs.max() - layer_outputs.min())
        if divider == 0:
            return np.zeros(shape=layer_outputs.shape)
        X_std = (layer_outputs - layer_outputs.min()) / divider
        X_scaled = X_std * (rmax - rmin) + rmin
        return X_scaled


    def predict(self, input_data):
        outputs = self.functor([input_data])
        return outputs


    def kmnc_update_coverage(self, ptr, layer_name, neuron_idx, idx, output):
        profiling_data_list = self.profiling_dict[(layer_name, neuron_idx)]
        #print(layer_name, neuron_idx)
        #print(self.profiling_dict)
        lower_bound = profiling_data_list[3]
        upper_bound = profiling_data_list[4]

        unit_range = (upper_bound - lower_bound) / self.k[0]

        # the special case, that a neuron output profiling is a fixed value
        # TODO: current solution see whether test data cover the specific value
        # if it covers the value, then it covers the entire range by setting to all 1s
        if unit_range == 0:
            return
        # we ignore output cases, where output goes out of profiled ranges,
        # this could be the surprised/exceptional case, and we leave it to
        # neuron boundary coverage criteria
        if output > upper_bound or output < lower_bound:
            return

        subrange_index = int((output - lower_bound) / unit_range)

        if subrange_index == self.k[0]:
            subrange_index -= 1

        # print "subranges: ", subrange_index

        id = self.start + self.layer_start_index[0][idx] + neuron_idx * self.bytearray_len[0] + subrange_index
        num = ptr[0][id]
        assert(num==0)
        if num < 255:
            num += 1
            ptr[0][id] = num


    # rev = True performs tknc
    def bknc_update_coverage(self, ptr, idx, layer_output_dict, rev):
        # sort the dict entry order by values
        sorted_index_output_dict = OrderedDict(
            sorted(layer_output_dict.items(), key=lambda x: x[1], reverse=rev))

        # for list if the top_k > current layer neuron number,
        # the whole list would be used, not out of bound
        top_k_node_index_list = list(sorted_index_output_dict.keys())[:self.k[1]]

        for top_sec, top_idx in enumerate(top_k_node_index_list):
            id = self.start + self.layer_start_index[1][idx] + top_idx * self.bytearray_len[1] + top_sec
            num = ptr[1][id]
            if num < 255:
                num += 1
                ptr[1][id] = num


    def nbc_snac_update_coverage(self, ptr, layer_name, neuron_idx, idx, output, nbc):
        profiling_data_list = self.profiling_dict[(layer_name, neuron_idx)]

        lower_bound = profiling_data_list[3]
        upper_bound = profiling_data_list[4]

        # this version uses k multi_section as unit range, instead of sigma
        # TODO: need to handle special case, std=0
        # TODO: this might be moved to args later
        k_multisection = 1000
        unit_range = (upper_bound - lower_bound) / k_multisection
        if unit_range == 0:
            unit_range = 0.05

        if nbc:
            # the hypo active case, the store targets from low to -infi
            if output < lower_bound:
                # float here
                target_idx = (lower_bound - output) / unit_range

                if target_idx > (self.k[2] - 1):
                    id = self.start + self.layer_start_index[2][idx] + neuron_idx * self.bytearray_len[2]\
                         + self.k[2] - 1
                else:
                    id = self.start + self.layer_start_index[2][idx] + neuron_idx * self.bytearray_len[2]\
                         + int(target_idx)

                num = ptr[2][id]
                if num < 255:
                    num += 1
                    ptr[2][id] = num
                return

        # the hyperactive case
        if output > upper_bound:
            target_idx = (output - upper_bound) / unit_range

            if nbc:
                if target_idx > (self.k[2] - 1):
                    id = self.start + self.layer_start_index[2][idx] + neuron_idx * self.bytearray_len[2]\
                         + self.k[2] - 1
                else:
                    id = self.start + self.layer_start_index[2][idx] + neuron_idx * self.bytearray_len[2] + int(
                        target_idx)
                num = ptr[2][id]
            else:
                if target_idx > (self.k[3] - 1):
                    id = self.start + self.layer_start_index[3][idx] + neuron_idx * self.bytearray_len[3] \
                         + self.k[3] - 1
                else:
                    id = self.start + self.layer_start_index[3][idx] + neuron_idx * self.bytearray_len[3] + int(
                        target_idx)
                num = ptr[3][id]
            if num < 255:
                num += 1
                if nbc:
                    ptr[2][id] = num
                else:
                    ptr[3][id] = num
            return

    def nc_update_coverage(self, ptr, layer_output, idx):
        '''
        Given the input, update the neuron covered in the model by this input.
            This includes mark the neurons covered by this input as "covered"
        :param input_data: the input image
        :return: the neurons that can be covered by the input
        '''
        scaled = self.scale(layer_output)
        for neuron_idx in range(scaled.shape[-1]):
            if np.mean(scaled[..., neuron_idx]) > self.k[4]:
                id = self.start + self.layer_start_index[4][idx] + neuron_idx * self.bytearray_len[4] + 0
                ptr[4][id] = 1
                #print(idx, neuron_idx)
                #print(self.nccsvArr)

                self.nccsvArr[idx][neuron_idx] += 1


    # overall coverage updating function
    def update_coverage(self, outputs):
        '''
        We implement the following metrics:
        NC from DeepXplore and DeepTest
        KMNC, BKNC, TKNC, NBC, SNAC from DeepGauge2.0.

        :param outputs: The outputs of internal layers for a batch of mutants
        :return: ptr is the array that record the coverage information
        '''

        ptr0 = np.zeros(self.total_size[0], dtype=np.uint8)
        ptr1 = np.zeros(self.total_size[1], dtype=np.uint8)
        ptr2 = np.zeros(self.total_size[2], dtype=np.uint8)
        ptr3 = np.zeros(self.total_size[3], dtype=np.uint8)
        ptr4 = np.zeros(self.total_size[4], dtype=np.uint8)
        ptr = [ptr0, ptr1, ptr2, ptr3, ptr4]

        if len(outputs) > 0 and len(outputs[0]) > 0:

            # Updates to be made:
            # (1) Currently, the code works with criteria 'all'; the logic should work for other criteria, but needs
            #       to be verified because the total coverage calculation may not function as intended (the coverage
            #       array for the other unused criteria may be filled in some weird way).
            # (2) NBC and SNAC are very similar, so their updating functions are combined into one. It would be useful
            #       to create a parameter that can choose to update the coverage arrays for both NBC and SNAC at the
            #       same time (in the same function call).
            # (3) Notice that in the code below, seed_id appears to only ever be 0. As such, that loop can essentially
            #       be removed.

            if self.criteria == 'all':
                for idx, layer_name in enumerate(self.layer_to_compute):
                    layer_outputs = outputs[idx]
                    # notice that seed_id is only 0
                    for seed_id, layer_output in enumerate(layer_outputs):
                        layer_output_dict = {}
                        for neuron_idx in range(layer_output.shape[-1]):
                            output = np.mean(layer_output[..., neuron_idx])
                            layer_output_dict[neuron_idx] = output
                            # kmnc
                            self.kmnc_update_coverage(ptr, layer_name, neuron_idx, idx, output)
                            # nbc --> to be combined better with snac, use a parameter "both" --> see update (2) above
                            self.nbc_snac_update_coverage(ptr, layer_name, neuron_idx, idx, output, True)
                            # snac
                            self.nbc_snac_update_coverage(ptr, layer_name, neuron_idx, idx, output, False)
                        # tknc
                        self.bknc_update_coverage(ptr, idx, layer_output_dict, True)
                        # nc
                        self.nc_update_coverage(ptr, layer_output, idx)

            # kmnc: 0
            elif self.criteria == 'kmnc':
                for idx, layer_name in enumerate(self.layer_to_compute):
                    layer_outputs = outputs[idx]
                    # notice that seed_id is only 0
                    for seed_id, layer_output in enumerate(layer_outputs):
                        for neuron_idx in range(layer_output.shape[-1]):
                            output = np.mean(layer_output[..., neuron_idx])
                            self.kmnc_update_coverage(ptr, layer_name, neuron_idx, idx, output)

            # tknc: 1
            elif self.criteria == 'tknc':
                for idx, layer_name in enumerate(self.layer_to_compute):
                    layer_outputs = outputs[idx]
                    # notice that seed_id is only 0
                    for seed_id, layer_output in enumerate(layer_outputs):
                        layer_output_dict = {}
                        for neuron_idx in range(layer_output.shape[-1]):
                            output = np.mean(layer_output[..., neuron_idx])
                            layer_output_dict[neuron_idx] = output
                        self.bknc_update_coverage(ptr, idx, layer_output_dict, True)

            # nbc: 2
            elif self.criteria == 'nbc':
                for idx, layer_name in enumerate(self.layer_to_compute):
                    layer_outputs = outputs[idx]
                    # notice that seed_id is only 0
                    for seed_id, layer_output in enumerate(layer_outputs):
                        for neuron_idx in range(layer_output.shape[-1]):
                            output = np.mean(layer_output[..., neuron_idx])
                            self.nbc_snac_update_coverage(ptr, layer_name, neuron_idx, idx, output, True)

            # snac: 3
            elif self.criteria == 'snac':
                for idx, layer_name in enumerate(self.layer_to_compute):
                    layer_outputs = outputs[idx]
                    # notice that seed_id is only 0
                    for seed_id, layer_output in enumerate(layer_outputs):
                        for neuron_idx in range(layer_output.shape[-1]):
                            output = np.mean(layer_output[..., neuron_idx])
                            self.nbc_snac_update_coverage(ptr, layer_name, neuron_idx, idx, output, False)

            # nc: 4
            elif self.criteria == 'nc':
                for idx, layer_name in enumerate(self.layer_to_compute):
                    layer_outputs = outputs[idx]
                    for seed_id, layer_output in enumerate(layer_outputs):
                        self.nc_update_coverage(ptr, layer_output, idx)

            else:
                print("* please select the correct coverage criteria as feedback:")
                print("['all', 'nc', 'kmnc', 'nbc', 'snac', 'bknc', 'tknc']")
                sys.exit(0)

        return ptr


if __name__ == '__main__':
    print("main Test.")