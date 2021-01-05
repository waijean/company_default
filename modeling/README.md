# Directory

| Folders   | Description |  
|:-------- |:----------- |
| config     |  string constants and final model hyperparameters  |
| experiments|  code used to run experiments |
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
