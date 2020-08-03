"""
Auxiliary script to generate several grid yaml files based from a single one with multiple options.


The script is meant to be used with snakemake, such that snakemake runs one parameter combination
for each file.
"""
import hashlib
import os

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import ParameterGrid
import argparse


def prepare_params_yaml(param_dic):
    """
    This function takes a single parameter dictionary and transforms it for
    the standard use in the network model.

    Parameters:
    ----------
    param_dic: dict, dictionary with params.
    """
    new_params = {}

    # init dictionaries in the main dictioarny
    for i in param_dic.keys():
        new_params[i.split("_")[0]] = {}

    # fill the dictionaries
    for i in param_dic.keys():
        new_params[i.split("_")[0]]["_".join(i.split("_")[1:])] = param_dic[i]

    return new_params


def is_valid_file(parser, arg):
    """
    Check if file exists.

    Args:
        parser: parse obj
        arg:  argument

    Returns:
        Error or File handle.
    """
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')  # return an open file handle


def transform_parameter_dic(params):
    """
    Extracts the grid parameters from the yaml file and saves them in a new
    dictionary without the other parameters.

    Parameters:
    ----------
    params: dict, dictionary with a single parameter set.
    """
    new_dic = {}
    for param in params["grid_parameters"].keys():
        for param_i in params["grid_parameters"][param].keys():
            new_dic[param + "_" + param_i] = \
                params["grid_parameters"][param][param_i]

    return list(ParameterGrid(new_dic))


def get_hash(in_str):
    """
    Generate a hash string from the input parameters.

    Parameters
    ----------
    in_str

    Returns
    -------

    """
    hash_object = hashlib.md5(bytes(in_str, encoding='utf-8'))
    return hash_object.hexdigest()


parser = argparse.ArgumentParser(
    description="xirt-grid - a handy helper for generating a set of hyperparameter definitions"
                "from a predefined yaml grid.")
parser.add_argument("-i", dest="infile", required=True,
                    help="YAML grid file holding parameter combinations.",
                    metavar="CSMS", type=lambda x: is_valid_file(parser, x))
parser.add_argument("-o", dest="outdir", required=True,
                    help="Output dir, where to store the individual yaml files and overview table.",
                    metavar="OUTDIR", type="str")
args = parser.parse_args()

outdir = args.outdir
infile = args.infile
#%%
infile = r"C:\\Users\\hanjo\\PycharmProjects\\xiRT\\sample_data\\xirt_grid.yaml"
outdir = "results/parameters/"

# create outdir
if not os.path.exists(outdir):
    os.makedirs(outdir)

print("Reading Parameter file...")
params = transform_parameter_dic(yaml.load(open(infile), Loader=yaml.FullLoader))

print("Collectin Parameter Info in df...")
parameters_ar = [pd.json_normalize(i) for i in params]
params_df = pd.concat(parameters_ar)
params_df["PARAM_ID"] = np.arange(len(params_df))
params_df["HASH"] = [get_hash(str(i)) for i in params]

print("Summary:")
print("Parameters: {}".format(params_df.shape[0]))
print("Unique hashes: {}".format(len(np.unique(params_df["HASH"]))))
params_df.set_index("PARAM_ID", drop=False, inplace=True)

print("Writing Parameter YAML files to: {}".format(outdir))
for param_i, hash_i in zip(params, params_df["HASH"]):
    param = prepare_params_yaml(param_i)
    with open('{}/{}.yaml'.format(outdir, hash_i), 'w') as outfile:
        yaml.dump(param, outfile, default_flow_style=False)

params_df.to_csv("{}/parameters.csv".format(outdir))
params_df.to_excel("{}/parameters.xls".format(outdir))
print("Done.")