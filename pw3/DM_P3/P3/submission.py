import json
import numpy as np
import os
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
import sys
import graderUtil

##########################################################
# Do not make any change in this block
# a dict stores the final result
task_result = {
    "y_pred": []
} 

# read task file content
training_filename = sys.argv[1]
testing_filename = sys.argv[2]
answer_filename = sys.argv[3]

df = graderUtil.load_file(training_filename)
X_test = graderUtil.load_testing_file(testing_filename)
y_test = graderUtil.load_answer_file(answer_filename)

print(df.sample(5))
print(X_test.sample(5))
print(y_test)
#
##########################################################
# BEGIN_YOUR_CODE


# prediction
task_result["y_pred"] = list(np.random.randint(2, size=y_test.shape[0]))

# END_YOUR_CODE
# Do Not Make Any Change BELOW
#######################################################################

# output your final result
print("Prediction:")
print(task_result["y_pred"])
print("Score(100%): " + str(graderUtil.accuracy_score(task_result['y_pred'],y_test['target'])))