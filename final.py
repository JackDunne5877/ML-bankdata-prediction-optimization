import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from statsmodels.api import OLS
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix

customers = pd.read_csv("final_train.csv")
customers.info()

corr_matrix = customers.corr()

customerCopy = customers.copy()


#turning data into binaries and numerical values

#Offer Accepted column: Yes becomes 1, No becomes 0
customerCopy["Offer Accepted"] = pd.Series(np.where(customerCopy["Offer Accepted"] == "Yes", 1, 0), customerCopy.index)

#Reward column: Air Miles become 0, Points become 1, cash back becomes 2
customerCopy["Reward"] = pd.Series(np.where(customerCopy["Reward"] != "Air Miles", customerCopy["Reward"], int(0)), customerCopy.index)
customerCopy["Reward"] = pd.Series(np.where(customerCopy["Reward"] != "Points", customerCopy["Reward"], int(1)), customerCopy.index)
customerCopy["Reward"] = pd.Series(np.where(customerCopy["Reward"] != "Cash Back", customerCopy["Reward"], int(2)), customerCopy.index)

#Mailer Type column: Letter becomes 0, postcard becomes 1
customerCopy["Mailer Type"] = pd.Series(np.where(customerCopy["Mailer Type"] == "Letter", 1, 0), customerCopy.index)

#Income level column: Low becomes 0, Medium becomes 1, High becomes 2
customerCopy["Income Level"] = pd.Series(np.where(customerCopy["Income Level"] != "Low", customerCopy["Income Level"], 0), customerCopy.index)
customerCopy["Income Level"] = pd.Series(np.where(customerCopy["Income Level"] != "Medium", customerCopy["Income Level"], 1), customerCopy.index)
customerCopy["Income Level"] = pd.Series(np.where(customerCopy["Income Level"] != "High", customerCopy["Income Level"], 2), customerCopy.index)

#Overdraft Protection column: Yes becomes 1, No becomes 0
customerCopy["Overdraft Protection"] = pd.Series(np.where(customerCopy["Overdraft Protection"] == "Yes", 1, 0), customerCopy.index)

#Credit Rating column: Low becomes 0, Medium becomes 1, High becomes 2
customerCopy["Credit Rating"] = pd.Series(np.where(customerCopy["Credit Rating"] != "Low", customerCopy["Credit Rating"], 0), customerCopy.index)
customerCopy["Credit Rating"] = pd.Series(np.where(customerCopy["Credit Rating"] != "Medium", customerCopy["Credit Rating"], 1), customerCopy.index)
customerCopy["Credit Rating"] = pd.Series(np.where(customerCopy["Credit Rating"] != "High", customerCopy["Credit Rating"], 2), customerCopy.index)

#Own your home column: Yes becomes 1, No becomes 0
customerCopy["Own Your Home"] = pd.Series(np.where(customerCopy["Own Your Home"] == "Yes", 1, 0), customerCopy.index)

customerCopy["Reward"].astype(object).astype(int)
corr_matrix2 = customerCopy.corr()

offerCorr = corr_matrix2["Offer Accepted"].sort_values(ascending=True)

attributes = ["Mailer Type", "Income Level", "# Bank Accounts Open", "Overdraft Protection", "Credit Rating",
              "# Credit Cards Held", "# Homes Owned", "Household Size", "Own Your Home", "Average Balance",]

attributes2 = ["Offer Accepted", "Income Level", "# Bank Accounts Open", "Average Balance", "Own Your Home"]

scatter_matrix(customerCopy[attributes2], figsize=(12, 8))
scatter_matrix(customerCopy[attributes], figsize=(12, 8))

customerCopy.plot(kind="scatter", x="Income Level", y="Average Balance")

#drop datapoints that could cause unwanted bias and colinearity
customerCopy = customerCopy.select_dtypes(include=[np.number])

train_set, test_set = train_test_split(customerCopy, test_size=0.25, random_state=380)

label = train_set["Offer Accepted"].copy()
features = train_set.drop("Offer Accepted", axis=1)
features = features.drop("Customer Number", axis=1)
features = features.drop("Average Balance", axis=1)

test_label = test_set["Offer Accepted"].copy()
test_features = test_set.drop("Offer Accepted", axis=1)
test_features = test_features.drop("Customer Number", axis=1)
test_features = test_features.drop("Average Balance", axis=1) #to eliminate colinearity
features.info()


imputer = SimpleImputer(strategy="median")
x = imputer.fit_transform(features)
features = pd.DataFrame(x, columns=features.columns, index=features.index)
features.info()

imputer2 = SimpleImputer(strategy="median")
x2 = imputer2.fit_transform(test_features)
test_features = pd.DataFrame(x2, columns=test_features.columns, index=test_features.index)
test_features.info()

#run a linear regression
lin_reg = LinearRegression()
lin_reg.fit(features, label)

#plot results
def check_result(observation, prediction):
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(1, 1, 1)
  ax.scatter(observation, prediction)
  ax.plot([0, 1], [0, 1], color='red')
  ax.axhline(y=0, color='k')
  ax.axvline(x=0, color='k')
  ax.set_ylim(0, prediction.max())
  ax.set_xlabel("observed value")
  ax.set_ylabel("predicted value")
  plt.show()

def accuracy_check(actualOutcome, predictedOutcome):
    total = 0
    counter = 0
    actualOutcome = actualOutcome.array
    for x in actualOutcome:
        total += 1
        if(actualOutcome[x] == 1 and predictedOutcome[x] >= 0.5
           or actualOutcome[x] == 0 and predictedOutcome[x] < 0.5):
            counter += 1
    print("test got " + str(counter) + " out of " + str(total))
    #return "test got %", counter and " right out of %", total
 
def yes_counter(actualOutcome, predictedOutcome):
    total = 0
    yes_counter = 0
    predicted_yes = 0
    actualOutcome = actualOutcome.array
    for x in actualOutcome:
        total += 1
        if(actualOutcome[x] == 1):
            yes_counter += 1
            if(predictedOutcome[x] == 1):
                predicted_yes += 1
    print("out of " + str(total) + ", " + str(yes_counter) + " were actual yeses, and " + str(predicted_yes) + "were predicted")


customer_predict_linear = lin_reg.predict(test_features)
customer_predict_linear = pd.Series(customer_predict_linear)
check_result(test_label, customer_predict_linear)
accuracy_check(test_label, customer_predict_linear)

#running forest regression method
forest_reg = RandomForestRegressor(n_estimators=30, random_state=380)
forest_reg.fit(features, label)
forest_prediction = forest_reg.predict(features)
check_result(label, forest_prediction)


test_predict_forest = forest_reg.predict(test_features)
check_result(test_label, test_predict_forest)
accuracy_check(test_label, test_predict_forest)


#run logistic regression
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=0)
clf = LogisticRegression()
clf.fit(x_train, y_train)
log_prediction = clf.predict(x_test)
clf = pd.Series(log_prediction)
check_result(y_test, log_prediction)
accuracy_check(y_test, log_prediction)
yes_counter(y_test, log_prediction)

def check_log_result(outcome, prediction):
    cm = confusion_matrix(outcome, prediction)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
    ax.set_ylim(1.5, -0.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
    plt.show()

check_log_result(y_test, clf)

print(OLS(label, features).fit().summary())

#trying out a binary logistic regression
"""
#SMOTE (sythetic minority oversampling technique) algorithm to create synthetic samples from
#the split with fewer observations (test)
X = features
y = label

from imblearn.over_sampling import SMOTE

os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['y']==0]))
print("Number of subscription",len(os_data_y[os_data_y['y']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))

columns = ["Mailer Type", "# Bank Accounts Open", "Overdraft Protection", "# Credit Cards Held",
           "# Homes Owned", "Household Size", "Own Your Home", "Q1 Balance",
           "Q2 Balance", "Q3 Balance", "Q4 Balance"]

features2 = os_data_X[columns]
label2 = os_data_y[label]

import statsmodels.api as sm
logit_model = sm.Logit(label, features)
result=logit_model.fit()
print(result.summary2())

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(features2, label2, test_size=0.2, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
"""


#actually predicting test results

test_customers = pd.read_csv("final_test.csv")
testCopy = test_customers.copy()
testCopy["Overdraft Protection"] = pd.Series(np.where(testCopy["Overdraft Protection"] == "Yes", 1, 0), testCopy.index)
testCopy["Own Your Home"] = pd.Series(np.where(testCopy["Own Your Home"] == "Yes", 1, 0), testCopy.index)
testCopy["Mailer Type"] = pd.Series(np.where(testCopy["Mailer Type"] == "Letter", 1, 0), testCopy.index)

customer_num = testCopy["Customer Number"].copy()


testCopy = testCopy.select_dtypes(include=[np.number])
testCopy = testCopy.drop("Customer Number", axis=1)
testCopy = testCopy.drop("Average Balance", axis=1)

test_features2 = testCopy.copy()
imputer3 = SimpleImputer(strategy="median")
x3 = imputer3.fit_transform(test_features2)
test_features2 = pd.DataFrame(x3, columns=test_features2.columns, index=test_features2.index)
test_features2.info()

#running forest regression method
forest_reg = RandomForestRegressor(n_estimators=30, random_state=380)
forest_reg.fit(features, label)
forest_prediction = forest_reg.predict(test_features2)

outcome_arr = [[]]
  
for x in range(0, len(forest_prediction) - 1):
    if(forest_prediction[x] >= 0.25):
        outcome_arr.insert(x, int(1))
    else:
        outcome_arr.insert(x, int(0))
   
#convert the outcome_arr list to a dataframe, with customer # correlated with relevant outcome
from pandas import DataFrame
output_df = DataFrame(customer_num, columns=["Customer Number"])

outcome_arr = np.array(outcome_arr)
output_df["Offer Accepted"] = outcome_arr

compression = dict(method='zip', archive_name='final_submission.csv')
output_df.to_csv('final_submission.zip', index=False, compression=compression)

