from attacks.targets.mnistMLP import trainMNISTMLP
from attacks.targets.cifar10ResNet import trainCIFAR10ResNet
from attacks.targets.mnistConvNet import trainMNISTConvNet
from attacks.utils import getDevice

import torch


import argparse

def main():
	prog = "build_target_models"
	descr = "Build target models for attack testing purposes"
	parser = argparse.ArgumentParser(prog=prog, description=descr)
	parser.add_argument("-m", "--model", type=str, default=None, required=False, help="Target Model")
	#parser.add_argument("", metavar="BINARIES", type=str, nargs="+", help="PE files to classify")
	args = parser.parse_args()

	targetModels = {"mnistMLP":trainMNISTMLP,"mnistConvNet":trainMNISTConvNet,"cifar10ResNet":trainCIFAR10ResNet}

	if not (args.model in targetModels.keys() or args.model is None):
		model_parse_error = "{} is not a supported model, try:\n".format(args.model)
		for m in targetModels.keys():
			model_parse_error += "\t" + m + "\n"
		parser.error(model_parse_error)

	
	
	if args.model is None:
		print("Training all of them")
		for model in targetModels.values():
			model(device)
	else:
		targetModels[args.model](device)


if __name__ == "__main__":
	main()