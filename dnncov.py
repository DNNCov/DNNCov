import shutil
import time
from mcdc import coverage_report_mcdc
import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import argparse
def create_directories(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    os.makedirs(dir+"/tests")
    os.makedirs(dir+"/queue")
    os.makedirs(dir+"/crashes")



parser = argparse.ArgumentParser(description='Optional app description')
#parser.add_argument('model', type=str,
#                    help='Specify Model (LENET1, LENET4, LENET5, TAXINET)')
#parser.add_argument('allcoverage', type=str,
#                    help='Specify True if you want to calculate KMNC, TKNC, NBC, SNAC and NC Coverage')
#parser.add_argument('mcdccoverage', type=str,
#                    help='Specify True if you want to calculate MCDC Coverage')
parser.add_argument('--train', type=int,
                    help='Specify # of Train Inputs used for MCDC',default=100,required=False)
parser.add_argument('--test', type=int,
                    help='Specify # of Test Inputs used for MCDC',default=100,required=False)

parser.add_argument(
        '--model', dest='model', default='-1', help='the input neural network model')

parser.add_argument("--criteria", dest="criteria", default=['nc', 'kmnc', 'snac', 'nbc', 'bknc', 'tknc', 'mcdc'],
                    help="the coverage metrics (NC, KMNC, SNAC, NBC, MCDC)", metavar="" , nargs='+')

parser.add_argument("--mnist-dataset", dest="mnist", help="MNIST dataset", action="store_true")
parser.add_argument("--normalized-input", dest="normalized", help="To normalize the input", action="store_true")
parser.add_argument("--cifar10-dataset", dest="cifar10", help="CIFAR-10 dataset", action="store_true")
parser.add_argument("--tinytaxinet-dataset", dest="tinytaxinet", help="CIFAR-10 dataset", action="store_true")
parser.add_argument("--outputs", dest="outputs", default="outs",
                    help="the outputput test data directory", metavar="DIR")

parser.add_argument("--input-rows", dest="img_rows", default="224",
                    help="input rows", metavar="INT")
parser.add_argument("--input-cols", dest="img_cols", default="224",
                    help="input cols", metavar="INT")
parser.add_argument("--input-channels", dest="img_channels", default="3",
                    help="input channels", metavar="INT")

parser.add_argument("--input-tests", dest="input_tests", default=None,
                    help="the test inputs directory", metavar="DIR")

parser.add_argument("--input-train", dest="input_train", default=None,
                    help="the train inputs directory", metavar="DIR")



args = parser.parse_args()
print(args)

create_directories(args.outputs)

dataset = ''
if args.mnist: dataset = 'mnist'
elif args.cifar10: dataset = 'cifar'
elif args.tinytaxinet: dataset = 'tinytaxinet'
elif not args.input_tests==None: dataset = 'inputs ' + args.input_tests

if (args.criteria[0]=='no-mcdc' or args.criteria[0]=='all'):
    start_time_main=time.time()
    subprocess.call("python ConstructTestSet.py -output_path {0}/test --{1} --input-rows {2} --input-cols {3} --input-channels {4}".format(args.outputs, dataset, args.img_rows, args.img_cols, args.img_channels), shell=True)
    subprocess.call("python Profile.py -model {0} -output_path {2}    --{1} --input-rows {3} --input-cols {4} --input-channels {5}".format(args.model, dataset, args.outputs, args.img_rows, args.img_cols, args.img_channels), shell=True)
    subprocess.call("python coverage_computer.py -model {1} -output_path {0} --{2} -i {0}/test".format(args.outputs, args.model, dataset), shell=True)
    print("Total Time for 5 Coverage Criteria: %s seconds" % round(time.time()-start_time_main,2))
img_dims = [(int)(args.img_rows), (int)(args.img_cols), (int)(args.img_channels)]
if not (args.input_tests==None): 
    dataset = [args.input_tests, args.input_train]
if (args.criteria[0]=='mcdc' or args.criteria[0]=='all'):
    start_time_main=time.time()
    coverage_report_mcdc(dataset,args.train,args.test, args.model, img_dims)
    print("Total Time for MCDC: %s seconds" % round(time.time()-start_time_main,2))

