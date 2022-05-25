import shutil

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
parser.add_argument('model', type=str,
                    help='Specify Model (LENET1, LENET4, LENET5, TAXINET)')
parser.add_argument('allcoverage', type=str,
                    help='Specify True if you want to calculate KMNC, TKNC, NBC, SNAC and NC Coverage')
parser.add_argument('mcdccoverage', type=str,
                    help='Specify True if you want to calculate MCDC Coverage')
parser.add_argument('--train', type=int,
                    help='Specify # of Train Inputs used for MCDC',default=100,required=False)
parser.add_argument('--test', type=int,
                    help='Specify # of Test Inputs used for MCDC',default=100,required=False)



args = parser.parse_args()
print(args)
if (args.allcoverage=="TRUE"):
    if(args.model=="LENET1"):
        create_directories("lenet1")
        subprocess.call("python ConstructTestSet.py -model_type lenet1 -output_path lenet1/test", shell=True)
        subprocess.call("python Profile.py -model lenet1.h5 -train mnist -o lenet1", shell=True)
        subprocess.call("python coverage_computer.py -i lenet1/test -o lenet1 -model lenet1 -criteria all", shell=True)

    if (args.model == "LENET4"):
        create_directories("lenet4")
        subprocess.call("python ConstructTestSet.py -model_type lenet4 -output_path lenet4/test", shell=True)
        subprocess.call("python Profile.py -model lenet4.h5 -train mnist -o lenet4", shell=True)
        subprocess.call("python coverage_computer.py -i lenet4/test -o lenet4 -model lenet4 -criteria all", shell=True)

    if (args.model == "LENET5"):
        create_directories("lenet5")
        subprocess.call("python ConstructTestSet.py -model_type lenet5 -output_path lenet5/test", shell=True)
        subprocess.call("python Profile.py -model lenet5.h5 -train mnist -o lenet5", shell=True)
        subprocess.call("python coverage_computer.py -i lenet5/test -o lenet5 -model lenet5 -criteria all", shell=True)

    if (args.model == "RESNET20"):
        create_directories("resnet20")
        subprocess.call("python ConstructTestSet.py -model_type resnet20 -output_path resnet20/test", shell=True)
        subprocess.call("python Profile.py -model resnet20.h5 -train cifar -o resnet20", shell=True)
        subprocess.call("python coverage_computer.py -i resnet20/test -o resnet20 -model resnet20 -criteria all", shell=True)

    if (args.model == "TAXINET"):
        create_directories("tinytaxinet")
        subprocess.call("python ConstructTestSet.py -model_type tinytaxinet -output_path tinytaxinet/test", shell=True)
        subprocess.call("python Profile.py -model KJ_TaxiNet.nnet -train tinytaxinet -o tinytaxinet", shell=True)
        subprocess.call("python coverage_computer.py -i tinytaxinet/test -o tinytaxinet -model tinytaxinet -criteria all", shell=True)

if (args.mcdccoverage == "TRUE"):
    if (args.model == "LENET1"):
        coverage_report_mcdc("lenet1",args.train,args.test,"","")
    if (args.model == "LENET4"):
        coverage_report_mcdc("lenet4",args.train,args.test,"","")
    if (args.model == "LENET5"):
        coverage_report_mcdc("lenet5",args.train,args.test,"","")
    if (args.model == "RESNET20"):
        coverage_report_mcdc("resnet20",args.train,args.test,"","")
    if (args.model == "TAXINET"):
        coverage_report_mcdc("tinytaxinet",args.train,args.test,'taxinetdataset/data-train/','taxinetdataset/data-val/')