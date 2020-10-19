#    name: addNoise.py
#  author: molloykp (Oct 2019)
# purpose: Accept a numpy npy file and add Gaussian noise
#          Parameters:
#   usage: python3 addNoise.py --inputFile MNISTXtrain1.npy --sigma 0.5 --outputFile MNISTXtrain1_noisy.npy 

import numpy as np
from pa2framework import print_greyscale
import argparse

def parseArguments():
    parser = argparse.ArgumentParser(description='Add noise to image(s)')

    parser.add_argument('--inputFile', action='store',
                        dest='inputFile', default="", required=True,
                        help='matrix of images in npy format')
    parser.add_argument('--sigma', action='store',type=float,
                        dest='sigma', default="", required=True,
                        help='std dev used in generating noise')

    parser.add_argument('--outputFile', action='store',
                        dest='outputFile', default="", required=True,
                        help='file to store matrix with noise')

    return parser.parse_args()

def main():
    np.random.seed(1671)

    parms = parseArguments()

    inMatrix = np.load(parms.inputFile)

    print(inMatrix[0].shape)

    # matrix must be floating point to add values
    # from the Gaussian
    outMatrix = inMatrix.astype('float32')
    outMatrix += np.random.normal(0,parms.sigma,(inMatrix.shape))
    outMatrix = outMatrix.astype('int')

    # noise may have caused values to go outside their allowable
    # range
    outMatrix[outMatrix < 0] = 0
    outMatrix[outMatrix > 255] = 255

    print_greyscale(inMatrix[0], inMatrix[0], width=28, height=28)

    np.save(parms.outputFile, outMatrix)

if __name__ == '__main__':
    main()
    
    