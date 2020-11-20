# Companies Bankruptcy Prediction

## Summary

Our study uses **financial ratios** data of Polish companies to predict their default using machine-learning techniques. 

The dataset used in this study is obtained from UCI Machine Learning Repository. The data contains 64 financial ratios 
and corresponding class label that indicates bankruptcy status after 2 years. Of the 9792 companies analyzed in this 
study, 515 companies (5.26%) went into bankruptcy, whereas 9277 (94.74%) firms survived.

Given that the dataset only contains financial ratios, instead of raw financial figures, we attempted and successfully 
**reverse engineered** the **raw financial figures** from financial ratios.  

We find that ensemble techniques such as random forest provide the best results. Furthermore, we applied 
*SHAP (SHapley Additive exPlanations)* technique to explain the output of the model.

The best way to step through our work is to view the 
[notebooks](https://github.com/waijean/company_default/tree/master/notebooks). 

## Data

The source data is obtained from 
[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data)

We created a [data dictionary](https://drive.google.com/file/d/1_XEkCGvHlyAAPn-jbbRPPjdY78BRIham/view?usp=sharing) 
to map the given column names to financial ratios:

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for 
development and testing purposes. 

#### conda environment
To set up the conda environment, run:

```bash
conda env create -f environment.yml
conda activate company_default
```

If there is any additonal packages required, add to the yml file and run:

```bash
conda env update -f environment.yml --prune
```

To create the kernel for jupyter notebooks, run:

```bash
conda activate company_default
python -m ipykernel install --user --name company_default --display-name "Python (company_default)`
```

#### pre-commit

To set up pre-commit (which is used to run black before committing to Git), run:

```bash
pre-commit install 
```

Add the following line in the first line of your notebook to run black formatting on that notebook:
```bash
%load_ext nb_black
```

#### input/output (io)

To set up the io, create a data folder in the root directory, 
which should have the following structure:

- data/
    - input/
        - train.csv
        - test.csv
    - output/

The input folder contains train.csv and test.csv, 
while the output folder will have the pipeline output. 

| Tables   | Description |  
|:-------- |:----------- |
| train.csv|  Labelled dataset used to train the model |
| test.csv |  Unlablled dataset to make prediciton for |


