"""This module contains the script to run xiML."""
import argparse
import os
import sys

import pandas as pd

from xirt import basic


if __name__ == "__main__":
    print("Hello there")
    print(basic.add_numbers(10, 20))

    print("Running {}".format(sys.argv[0]))
    print("xiML config {}".format(sys.argv[1]))
    print("xiRT config {}".format(sys.argv[2]))
    print("Working Directory: {}".format(os.getcwd()))

    # parser = argparse.ArgumentParser(
    #     description='xiRT - Retention Time Prediction for Linear and Cross-Linked Peptides '
    #                 'in Mulitiple Dimensions.')
    # parser.add_argument('-c', action='store', dest='config', help='YAML configuration file.')