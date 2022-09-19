"""
This file creates the data for the dummy prediction.
It generates the csv file from then given link and,
stores that in the data folder.
By default if no path is given it directly sends the csv file as
binaries to the called function.
"""

import argparse
import logging
import os
import tarfile

import pandas as pd
from six.moves import urllib

parser = argparse.ArgumentParser()
parser.add_argument("--output_path",
                    help="Output Path of the data",
                    default=os.path.dirname(__file__))
parser.add_argument("--log-level",
                    help="Choose Log Level from the choice.", default='DEBUG',
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
parser.add_argument("--log-path",
                    help="Choose path for log storing", default='')
parser.add_argument("--no-console-log",
                    help="Write Logs to console", default='False')
args = parser.parse_args()

output_path = args.output_path
logLevel = args.log_level
log_path = args.log_path
no_console_log = args.no_console_log

logger = logging.getLogger()

if logLevel.upper() == 'CRITICAL':
    logging.basicConfig(level=logging.CRITICAL)
if logLevel.upper() == 'ERROR':
    logging.basicConfig(level=logging.ERROR)
if logLevel.upper() == 'INFO':
    logging.basicConfig(level=logging.INFO)
if logLevel.upper() == 'DEBUG':
    logging.basicConfig(level=logging.DEBUG)

if log_path != '':
    l1 = os.path.join(log_path, 'Logs.log')
    logging.basicConfig(filename=l1)

if no_console_log:
    logging.disable(logging.DEBUG)

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    This function retrieves the data-set from the given Url.

    Parameters
    ----------
    HOUSING_URL : str
        HOUSING URL.
    housing_path : str
        Housing path.

    Returns
    -------
    None

    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH, output_path=output_path):
    """
    This function stores the generated csv into the given folder.
    Default: Stores in data folder and passes the csv binaries to the
    called function.

    output_path: path of the out  folder e.g. : 'data/'

    This function retrieves the data-set from the given Url.

    Parameters
    ----------
    housing_path : str
        Housing path.
    output_path : str
        Output path for data generation.

    Returns
    -------
    output_path == ''
        None.
    output_path != ''
        csv file

    """
    csv_path = DOWNLOAD_ROOT + os.path.join(housing_path, "housing.csv")
    if output_path == '':
        pd.read_csv(csv_path).to_csv('input_data.csv')
        logging.debug('Data Set Created at: {}'.format("data/input_data.csv"))
    else:
        j = os.path.join(output_path, 'input_data.csv')
        pd.read_csv(csv_path).to_csv(j)
        logging.debug('Data Set Created at: {}'.format(j))
    return pd.read_csv(csv_path)
