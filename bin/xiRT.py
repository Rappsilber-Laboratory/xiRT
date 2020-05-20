"""
This module contains the script to run xiML.
"""
from xirt import basic
import sys
import os
import pandas as pd

def preprocess():
    psms_df = pd.read_csv(psms_loc)


if __name__ == "__main__":
    print("Hello there")
    print(basic.add_numbers(10, 20))


    print("Running {}".format(sys.argv[0]))
    print("xiML config {}".format(sys.argv[1]))
    print("xiRT config {}".format(sys.argv[2]))
    print("Working Directory: {}".format(os.getcwd()))