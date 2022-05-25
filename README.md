
# DNNCov: An Integrated Coverage Measuring Tool for Deep Neural Networks

Getting Started Guide

## Instructions for Reproducing Results
- tensorflow==2.6.0
- keras==2.6.0
- numpy==1.19.5
- matplotlib==3.4.3
- h5py==3.1.0
- pillow==8.3.2
- scipy==1.7.1
- pandas==1.3.3
- requests==2.26.0

You can install one using pip package manager, e.g.:
   - pip3 install tensorflow keras pandas numpy
   
Datasets:
-	The tool will automatically download the MNIST and CIFAR10 datasets.  
-	For Tiny-TaxiNet Dataset, keep the taxinetdataset folder in the main directory. 
### Note:
All commands should be run inside DNN-Coverage directory.

## Run Command:

`python dnncov.py --model [lenet1.h5, lenet4.h5, lenet5.h5] --[mnist, cifar10, tinytaxinet]-dataset --outputs [output_directory] --criteria ["all", no-mcdc, mcdc] --train [#train for mcdc] --test[#test for mcdc]`

- \-\-model specifies the trained neural network to be tested. Possible options are: 
 	- lenet1.h5 
	- lenet4.h5 
	- lenet5.h5
	- resnet20.h5
	- tinytaxinet.h5
- \-dataset specifies the dataset to be used. 	
	- \-\-mnist-dataset specifies to use the MNIST dataset.
	- \-\-cifar10-dataset specifies to use the CIFAR10 dataset.
	- \-\-tinytaxinet-dataset specifies to use the TinyTaxiNet dataset.
- \-\-criteria specifies which coverage measures to calculate
	-  "all" to calculate KMNC, TKNC, NBC, SNAC, NC and MC/DC coverage. 
	-  "mcdc" to calculate only MC/DC coverage.
	-  "no-mcdc" to calculate KMNC, TKNC, NBC, SNAC and NC coverage. 
- \-\- outputs specifies the directory for storing the coverage report results.
- \-\-train specifies the # of Train inputs to be used for MCDC Coverage. This argument is optional. Default value is 100.
- \-\-test specifies the # of Test inputs to be used for MCDC Coverage. This argument is optional. Default value is 100.


### Example Commands For running only MC/DC Coverage:

#### LENET1:
`python dnncov.py --model lenet1.h5 --criteria mcdc --mnist-dataset --outputs lenet1output --train 10000 --test 10000`

#### LENET4:
`python dnncov.py --model lenet4.h5 --criteria mcdc --mnist-dataset --outputs lenet4output --train 10000 --test 10000`

#### LENET5:
`python dnncov.py --model lenet5.h5 --criteria mcdc --mnist-dataset --outputs lenet5output --train 10000 --test 10000`

#### RESNET20:
`python dnncov.py --model resnet20.h5 --criteria mcdc --cifar10-dataset --outputs taxinetoutput --train 10000 --test 10000`

#### TAXINET:
`python dnncov.py --model tinytaxinet.h5 --criteria mcdc --tinytaxinet-dataset --outputs taxinetoutput --train 10000 --test 10000`

### Example Commands For KMNC, TKNC, NBC, SNAC and NC Coverage:

#### LENET1:
`python dnncov.py --model lenet1.h5 --criteria no-mcdc --mnist-dataset --outputs lenet1output --train 10000 --test 10000`

#### LENET4:
`python dnncov.py --model lenet4.h5 --criteria no-mcdc --mnist-dataset --outputs lenet4output --train 10000 --test 10000`

#### LENET5:
`python dnncov.py --model lenet5.h5 --criteria no-mcdc --mnist-dataset --outputs lenet5output --train 10000 --test 10000`

#### RESNET20:
`python dnncov.py --model resnet20.h5 --criteria no-mcdc --cifar10-dataset --outputs taxinetoutput --train 10000 --test 10000`

#### TAXINET:
`python dnncov.py --model tinytaxinet.h5 --criteria no-mcdc --tinytaxinet-dataset --outputs taxinetoutput --train 10000 --test 10000`

## Evaluation Subjects
| Model       | Benchmark | #Dataset  (train,test) | Test Accuracy             | Model Architecture                                         |
|-------------|-----------|------------------------|---------------------------|------------------------------------------------------------|
| LeNet-1     | MNIST     | (60k,10k)              | 90.6%                     | 2 conv + 2 maxpool                                         |
| LeNet-4     | MNIST     | (60k,10k)              | 89.9%                     | 2 conv + 2 maxpool + 1 dense                               |
| LeNet-5     | MNIST     | (60k,10k)              | 92.8%                     | 2 conv + 2 maxpool + 2 dense                               |
| ResNet20    | CIFAR-10  | (50k,10k)              | 68.7%                     | 21 conv + 19 batchnorm + 19 act  + 9 add + 1 globalavgpool |
| TinyTaxiNet | TaxiNet   | (51462,7386)           | MAE  (cte: 1.44, he:2.75) | 3 dense                                                    |

