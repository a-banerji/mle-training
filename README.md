# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data.

The following techniques have been used:

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## To replicate python environment
conda env create -f env.yml
conda activate mle-dev

## To install dependencies
In the project directory "pip install . "

## to create data-csv
Inside the data folder type in terminal:
    python -c 'import ingest_data; ingest_data.load_housing_data()' --output_path=<path where you want the input_data.csv to be created>

Note: On default it stores on the data folder

## To run the script for training and storing the model weights:
python train.py < --input_data: "Input Path of the data" > < --model_output_path: "Model save path" >

Note: On default it reads input from data folder and stores output in pickles folder.

## To validate the modeel and store model scores:
python store.py < --model_load_path: "Input Path of the model" > < --data_load_path: "Data-set path" > <--output : "Output Excel path for scores">

Note: On default it reads model from pickles folder, data-set from data folder and stores the excel for scores in output folder.

## To install and run the pakage using wheel
Inside the dist folder: pip install mle_training-0.2-py3-none-any.whl

from data import ingest_data
from scripts import trainmodel

Pass the parameters inside the functions to run the tool.
