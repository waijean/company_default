# Directory

| Folders   | Description |  
|:-------- |:----------- |
| reverse_engineer |  code used to reverse enginner raw values from ratios  |
| tests      |  tests for utilities functions |
| utils      |  utilities functions |


### preprocessing runner

There are two versions of preprocessing runner: train and test. The train 
version will drop some training examples (e.g duplicates, too many null values) 
whereas the test version will retain all the examples since we need to make 
predictions for all the examples.

The preprocessing runner will output three files:
1. ratio
2. raw
3. combined 

The **combined** dataset is primarily used for downstream modeling purposes. The
reason we have the ratio and raw dataset separated is to validate whether 
reverse engineering the raw values helps to improve the model performance. 