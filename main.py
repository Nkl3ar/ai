# https://www.kaggle.com/datasets/prakashraushan/loan-dataset/data?select=LoanDataset+-+LoansDatasest.csv

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
pd.options.display.max_columns = 30
pd.options.display.max_rows = 151

keep_historical_default = True
# null je 64% podataka u historical default
historical_default_ignore_null = False
historical_default_null_value = 1
# N:0,Y:1

legal_child_labor_age = 14


# koristilo se prilikom odabira značajki
#normalize = False

#rows_to_norm = ['customer_age', 'customer_income',
#       'employment_duration',  'loan_grade', 'loan_amnt',
#       'loan_int_rate', 'term_years', 'cred_hist_length']

# home_ownership, Current_loan_status, loan_intent, loan_grade, historical_default mapirane vrijednosti

df = pd.read_csv("LoanDataset - LoansDatasest.csv", header=0)

df.drop("customer_id", axis=1, inplace=True)
#print(df.columns)

# sumnjivi podatci
df.drop(df[df.customer_age < 18].index, inplace=True)
df.drop(df[df.customer_age > 95].index, inplace=True)
#df.drop(df[df.employment_duration > 50].index, inplace=True)

# id 29161 ima "250,000" umjesto 250000
df['customer_income'] = df['customer_income'].str.replace(',', '')

for i in df.index:
    #print(df["customer_income"][i])
    #print(df["customer_age"][i])
    if df["customer_age"][i]-legal_child_labor_age < df["employment_duration"][i]:
        df.drop(i,inplace=True)
    elif df["customer_age"][i]-legal_child_labor_age < df["cred_hist_length"][i]:
        df.drop(i,inplace=True)

if keep_historical_default:
    if historical_default_ignore_null:
        hist_def_to_num = {'N': 0, 'Y': 1}
        df['historical_default'] = df['historical_default'].map(hist_def_to_num)
    else:
        hist_def_to_num = {np.nan: historical_default_null_value, 'N': 0, 'Y': 1}
        df['historical_default'] = df['historical_default'].map(hist_def_to_num)
else:
    df.drop("historical_default", axis=1, inplace=True)

df['loan_amnt'] = df['loan_amnt'].str.replace('£', '')
df['loan_amnt'] = df['loan_amnt'].str.replace(',', '')

home_own_to_num = {'OWN': 0,
                   'RENT': 1,
                   'MORTGAGE': 2,
                   'OTHER': 3}
df['home_ownership'] = df['home_ownership'].map(home_own_to_num)

loan_stat_to_num = {'NO DEFAULT': 0,
                    'DEFAULT': 1}
df['Current_loan_status'] = df['Current_loan_status'].map(loan_stat_to_num)

loan_intent_to_num = {'MEDICAL': 0,
                      'EDUCATION': 1,
                      'HOMEIMPROVEMENT': 2,
                      'DEBTCONSOLIDATION': 3,
                      'PERSONAL': 4,
                      'VENTURE': 5}
df['loan_intent'] = df['loan_intent'].map(loan_intent_to_num)

loan_grade_to_num = {"A": 0,
                     "B": 1,
                     "C": 2,
                     "D": 3,
                     "E": 4}
df['loan_grade'] = df['loan_grade'].map(loan_grade_to_num)

df.dropna(axis=0, how='any', inplace=True)

#if normalize:
#    scaler = StandardScaler()
#    normalized_data = scaler.fit_transform(df)
#    df_normalized = pd.DataFrame(normalized_data, columns=df.columns)
#    for i in rows_to_norm:
#        df[i] = df_normalized[i]


#print(df)
#print(df['loan_amnt'].unique())
print(df.describe())

df_corr = df.corr().abs().unstack().sort_values(ascending=True)
print(df_corr)


#Current_loan_status  loan_int_rate          0.340064
#Current_loan_status  loan_grade             0.369906
#Current_loan_status  historical_default     0.666722
print(df)

#plt.scatter(df["customer_age"], df["customer_income"])
#plt.scatter(df["Current_loan_status"], df["loan_int_rate"])
plt.xlabel("customer age")
plt.ylabel("customer income")
df_sorted = df.sort_values(by=['customer_income'], ascending=False)
plt.scatter(df_sorted["customer_age"], df_sorted["customer_income"])
plt.show()

plt.xlabel("customer age")
plt.ylabel("employment duration")
plt.scatter(df["customer_age"], df["employment_duration"])
plt.show()
#knn, svm, logreg


colX = ['loan_int_rate', 'loan_grade', 'historical_default']
X = df.loc[:, colX]
colY = ['Current_loan_status']
Y = df.loc[:, colY].values.ravel()
#X, Y = shuffle(X, Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.5, random_state=0)

ss = StandardScaler()
X_std_train = ss.fit_transform(X_train)
X_std_test = ss.fit_transform(X_test)

knn = KNeighborsClassifier(n_neighbors=20,metric='minkowski')
knn.fit(X_std_train, Y_train)
Y_test_pred_knn = cross_val_predict(knn, X_std_test, Y_test, cv=7)

svm = svm.SVC(kernel='linear', C=0.001)
svm.fit(X_std_train, Y_train)
Y_test_pred_svm = cross_val_predict(svm, X_std_test, Y_test, cv=7)

lr = LogisticRegression(C=1.0, penalty='l2', tol=0.0001, solver="lbfgs")
lr.fit(X_std_train, Y_train)
Y_test_pred_lr = cross_val_predict(lr, X_std_test, Y_test, cv=7)


print(confusion_matrix(Y_test, Y_test_pred_knn))

print("Precision Score: \t {0:.4f}".format(precision_score(Y_test,
                                                            Y_test_pred_knn,
                                                            average='weighted')))
print("Recall Score: \t\t {0:.4f}".format(recall_score(Y_test,
                                                      Y_test_pred_knn,
                                                      average='weighted')))
print("F1 Score: \t\t {0:.4f}".format(f1_score(Y_test,
                                              Y_test_pred_knn,
                                              average='weighted')))

print(confusion_matrix(Y_test, Y_test_pred_svm))

print("Precision Score: \t {0:.4f}".format(precision_score(Y_test,
                                                            Y_test_pred_svm,
                                                            average='weighted')))
print("Recall Score: \t\t {0:.4f}".format(recall_score(Y_test,
                                                      Y_test_pred_svm,
                                                      average='weighted')))
print("F1 Score: \t\t {0:.4f}".format(f1_score(Y_test,
                                              Y_test_pred_svm,
                                              average='weighted')))

print(confusion_matrix(Y_test, Y_test_pred_lr))

print("Precision Score: \t {0:.4f}".format(precision_score(Y_test,
                                                            Y_test_pred_lr,
                                                            average='weighted')))
print("Recall Score: \t\t {0:.4f}".format(recall_score(Y_test,
                                                      Y_test_pred_lr,
                                                      average='weighted')))
print("F1 Score: \t\t {0:.4f}".format(f1_score(Y_test,
                                              Y_test_pred_lr,
                                              average='weighted')))
