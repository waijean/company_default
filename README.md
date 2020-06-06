# company_default

The project's aim is to use company's financial ratios to predict whether a company will default in the next 5 months.
 
## Getting Started

These instructions will get you a copy of the project up and running on your local machine for 
development and testing purposes. See deployment for notes on how to deploy the project on a live 
system.

The Python version used is 3.7.1 and it is recommended to create a virtual environment and install
the required packages:  
```bash
#cd to root directory
python -m venv venv

source venv/Scripts/activate (Windows) 
source venv/bin/activate (Mac/Linux)

pip install -r requirements.txt
```

Next, pre-commit is used to run black and nbstripout before committing to Git. The 
.pre-commit-config.yaml file is included in the repo so only the following step is required after 
installing the pre-commit package (which is included in the requirements.txt):
```bash
pre-commit install 
```

To use the virtual environment in notebook, you have to install ipykernel package first (which is also
included in the requirements.txt). Then, you need to add your virtual environment to Jupyter:
```bash
python -m ipykernel install --user --name=venv
```

## Deployment

To deploy project on a live system
