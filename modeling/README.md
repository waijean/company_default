# Directory

| Folders   | Description |  
|:-------- |:----------- |
| config     |  string constants and final model hyperparameters  |
| experiments|  code used to run experiments |
| mlrun      |  mlflow code |
| utils      |  utilities functions |

### backtesting runner
The backtesting runner is used to run experiments by first splitting the data
into train and test set, and then further split the train set into 5-fold 
cross-validation set. 

The metrics are computed on both the cross-validation set and test set. 
However, we only tune the model based on cross-validation metrics. The test 
set metrics are only used to report the generalization errors.


### random search runner
The random search runner is used to perform hyperparameter tuning and 
give us the best hyperparameter based on the cross validation metrics.

Similar to backtesting runner, it will report the test set metrics but
this information is not used to tune the model.


### model runner
Once we have finalized the model, we train the model using the full dataset
to make predictions for unlabelled data. 

The model runner will write the submission file (which contains the prediction)
to the output folder.


### mlflow (optional)

*Note: While this tool will be useful to set up machine learning models in production and track data science experiments, 
we find this tool too restrictive when running experiments to explore different modeling approaches.* 

Steps to run MLflow:
1. Construct sklearn pipeline in modeling/mlrun/cross_validate_runner.py
2. This will log the params, metrics, tags, and artifacts into a specific run folder within mlruns directory. 
3. To view the results using mlflow UI, run the following command and view it at http://localhost:5000

```bash
#cd to root directory
mlflow ui
```
