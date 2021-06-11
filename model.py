# Python libraries
# Classic,data manipulation and linear algebra
import pandas as pd
import numpy as np

# Plots
import seaborn as sns
import matplotlib.pyplot as plt


# Data processing, metrics and modeling
import lightgbm as lgbm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, confusion_matrix,  roc_curve, precision_recall_curve, accuracy_score, roc_auc_score
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict


# Stats
import scipy.stats as ss
from scipy import interp
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

# Plots
import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.tools as tls
import plotly.figure_factory as ff

# To save model
import joblib


# read csv file diabetes_original.csv

db = pd.read_csv('diabetes_original.csv')

db[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = db[[
    'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
#Outcome	 | Glucose
#0	         | 107
#1	         | 140

db.loc[(db['Outcome'] == 0) & (db['Glucose'].isnull()), 'Glucose'] = 107
db.loc[(db['Outcome'] == 1) & (db['Glucose'].isnull()), 'Glucose'] = 140


# Outcome	 | Insulin
# 0	         | 102.5
# 1	         | 169.5


db.loc[(db['Outcome'] == 0) & (db['Insulin'].isnull()), 'Insulin'] = 102.5
db.loc[(db['Outcome'] == 1) & (db['Insulin'].isnull()), 'Insulin'] = 169.5


# Outcome	 | SkinThickness
# 0	         | 27
# 1	         | 32


db.loc[(db['Outcome'] == 0) & (
    db['SkinThickness'].isnull()), 'SkinThickness'] = 27
db.loc[(db['Outcome'] != 0) & (
    db['SkinThickness'].isnull()), 'SkinThickness'] = 32

# Outcome	 | BloodPressure
# 0	         | 70
# 1	         | 74.5


db.loc[(db['Outcome'] == 0) & (
    db['BloodPressure'].isnull()), 'BloodPressure'] = 70
db.loc[(db['Outcome'] != 0) & (
    db['BloodPressure'].isnull()), 'BloodPressure'] = 74.5

# Outcome	 | BMI
# 0	         | 30.1
# 1	         | 34.3

db.loc[(db['Outcome'] == 0) & (db['BMI'].isnull()), 'BMI'] = 30.1
db.loc[(db['Outcome'] != 0) & (db['BMI'].isnull()), 'BMI'] = 34.3

# Feature Engineering
# Creating new features out of existing ones

# Healthy people are concentrate with an age <= 30 and glucose <= 120
db.loc[:, 'N1'] = 0
db.loc[(db['Age'] <= 30) & (db['Glucose'] <= 120), 'N1'] = 1

# According to wikipedia "The body mass index (BMI) or Quetelet index is a value derived from the mass (weight) and height of an individual.
# The BMI is defined as the body mass divided by the square of the body height, and is universally expressed in units of kg/m2, resulting from
# mass in kilograms and height in metres."

# 30 kg/m² is the limit to obesity

db.loc[:, 'N2'] = 0
db.loc[(db['BMI'] <= 30), 'N2'] = 1

# Women with age <= 30 and Number of Pregnancies <= 6 are less likely to have diabetes

db.loc[:, 'N3'] = 0
db.loc[(db['Age'] <= 30) & (db['Pregnancies'] <= 6), 'N3'] = 1

# Healthy persons are concentrate with an blood pressure <= 80 and glucose <= 105

db.loc[:, 'N4'] = 0
db.loc[(db['Glucose'] <= 105) & (db['BloodPressure'] <= 80), 'N4'] = 1

# Healthy persons are concentrate with a skin thickness <= 20

db.loc[:, 'N5'] = 0
db.loc[(db['SkinThickness'] <= 20), 'N5'] = 1


# Healthy persons are concentrate with a BMI < 30 and skin thickness <= 20

db.loc[:, 'N6'] = 0
db.loc[(db['BMI'] < 30) & (db['SkinThickness'] <= 20), 'N6'] = 1

# Healthy person is concentrated with Glucose <= 105 and BMI <= 30

db.loc[:, 'N7'] = 0
db.loc[(db['Glucose'] <= 105) & (db['BMI'] <= 30), 'N7'] = 1

# Insulin level should be less than 200 for healthy body

db.loc[:, 'N9'] = 0
db.loc[(db['Insulin'] < 200), 'N9'] = 1

# Blood Pressure should be less than 80

db.loc[:, 'N10'] = 0
db.loc[(db['BloodPressure'] < 80), 'N10'] = 1

# with more number of pregnancies there is chance of having diabetes

db.loc[:, 'N11'] = 0
db.loc[(db['Pregnancies'] < 4) & (db['Pregnancies'] != 0), 'N11'] = 1

#  skin layers became progressively thicker with increasing BMI

db['N0'] = db['BMI'] * db['SkinThickness']

# Number of pregnancies per year

db['N8'] = db['Pregnancies'] / db['Age']

# Ratio of Glucose and Diabetes Pedigree Function

db['N13'] = db['Glucose'] / db['DiabetesPedigreeFunction']


db['N12'] = db['Age'] * db['DiabetesPedigreeFunction']

# Insulin levels varies with Age. So N14 is used to check the ratio of age to insulin to get the idea of the insulin levels at certain age

db['N14'] = db['Age'] / db['Insulin']

# N0 should be less than 1034 for healthy functioning

db.loc[:, 'N15'] = 0
db.loc[(db['N0'] < 1034), 'N15'] = 1


# Data Preparation for Machine Learnig Models

target_col = ["Outcome"]

cat_cols = db.nunique()[db.nunique() < 12].keys().tolist()

cat_cols = [x for x in cat_cols]
# numerical columns
num_cols = [x for x in db.columns if x not in cat_cols + target_col]
# Binary columns with 2 values
bin_cols = db.nunique()[db.nunique() == 2].keys().tolist()
# Columns more than 2 values
multi_cols = [i for i in cat_cols if i not in bin_cols]

# Label encoding Binary columns
le = LabelEncoder()
for i in bin_cols:
    db[i] = le.fit_transform(db[i])

# Duplicating columns for multi value columns
db = pd.get_dummies(db, columns=multi_cols)

# Scaling Numerical columns
std = StandardScaler()

scaled = std.fit_transform(db[num_cols])

scaled = pd.DataFrame(scaled, columns=num_cols)

# dropping original values merging scaled values for numerical columns
df_db_og = db.copy()
db = db.drop(columns=num_cols, axis=1)
db = db.merge(scaled, left_index=True, right_index=True, how="left")

X = db.drop('Outcome', 1)
y = db['Outcome']


# This function is to check the score of the model
# Classification model are evaluated on mainly Precision, Recall, F1 Score, 'ROC curve'

def scores_table(model, subtitle):
    scores = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    res = []
    for sc in scores:
        scores = cross_val_score(model, X, y, cv=5, scoring=sc)
        res.append(scores)
    df = pd.DataFrame(res).T
    df.loc['mean'] = df.mean()
    df.loc['std'] = df.std()
    df = df.rename(columns={0: 'accuracy', 1: 'precision',
                   2: 'recall', 3: 'f1', 4: 'roc_auc'})

# Visulalizing the accuracy, precision, recall, f1 and roc-auc

    trace = go.Table(
        header=dict(values=['<b>Fold', '<b>Accuracy', '<b>Precision', '<b>Recall', '<b>F1 score', '<b>Roc auc'],
                    line=dict(color='#7D7F80'),
                    fill=dict(color='#a1c3d1'),
                    align=['center'],
                    font=dict(size=15)),
        cells=dict(values=[('1', '2', '3', '4', '5', 'mean', 'std'),
                           np.round(df['accuracy'], 3),
                           np.round(df['precision'], 3),
                           np.round(df['recall'], 3),
                           np.round(df['f1'], 3),
                           np.round(df['roc_auc'], 3)],
                   line=dict(color='#7D7F80'),
                   fill=dict(color='#EDFAFF'),
                   align=['center'], font=dict(size=15)))

    layout = dict(width=800, height=400,
                  title='<b>Cross Validation - 5 folds</b><br>'+subtitle, font=dict(size=15))
    fig = dict(data=[trace], layout=layout)

    py.plot(fig, filename='styled_table')


def model_performance(model, subtitle):
    # Kfold
    cv = KFold(n_splits=5, shuffle=False, random_state=None)
    y_real = []
    y_proba = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 1

    for train, test in cv.split(X, y):
        model.fit(X.iloc[train], y.iloc[train])
        pred_proba = model.predict_proba(X.iloc[test])
        precision, recall, _ = precision_recall_curve(
            y.iloc[test], pred_proba[:, 1])
        y_real.append(y.iloc[test])
        y_proba.append(pred_proba[:, 1])
        fpr, tpr, t = roc_curve(y[test], pred_proba[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    # Confusion matrix
    y_pred = cross_val_predict(model, X, y, cv=5)
    conf_matrix = confusion_matrix(y, y_pred)
    trace1 = go.Heatmap(z=conf_matrix, x=["0 (pred)", "1 (pred)"],
                        y=["0 (true)", "1 (true)"], xgap=2, ygap=2,
                        colorscale='Viridis', showscale=False)

    # Show metrics
    tp = conf_matrix[1, 1]
    fn = conf_matrix[1, 0]
    fp = conf_matrix[0, 1]
    tn = conf_matrix[0, 0]
    Accuracy = ((tp+tn)/(tp+tn+fp+fn))
    Precision = (tp/(tp+fp))
    Recall = (tp/(tp+fn))
    F1_score = (2*(((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn)))))

    show_metrics = pd.DataFrame(data=[[Accuracy, Precision, Recall, F1_score]])
    show_metrics = show_metrics.T

    colors = ['gold', 'lightgreen', 'lightcoral', 'lightskyblue']
    trace2 = go.Bar(x=(show_metrics[0].values),
                    y=['Accuracy', 'Precision', 'Recall', 'F1_score'], text=np.round_(show_metrics[0].values, 4),
                    textposition='auto', textfont=dict(color='black'),
                    orientation='h', opacity=1, marker=dict(
        color=colors,
        line=dict(color='#000000', width=1.5)))

    # Roc curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)

    trace3 = go.Scatter(x=mean_fpr, y=mean_tpr,
                        name="Roc : ",
                        line=dict(color=('rgb(22, 96, 167)'), width=2), fill='tozeroy')
    trace4 = go.Scatter(x=[0, 1], y=[0, 1],
                        line=dict(color=('black'), width=1.5,
                        dash='dot'))

    # Precision - recall curve
    y_real = y
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)

    trace5 = go.Scatter(x=recall, y=precision,
                        name="Precision" + str(precision),
                        line=dict(color=('lightcoral'), width=2), fill='tozeroy')

    mean_auc = round(mean_auc, 3)
    # Subplots
    fig = tls.make_subplots(rows=2, cols=2, print_grid=False,
                            specs=[[{}, {}],
                                   [{}, {}]],
                            subplot_titles=('Confusion Matrix',
                                            'Metrics',
                                            'ROC curve'+" " +
                                            '(' + str(mean_auc)+')',
                                            'Precision - Recall curve',
                                            ))
    #Trace and layout
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)
    fig.append_trace(trace3, 2, 1)
    fig.append_trace(trace4, 2, 1)
    fig.append_trace(trace5, 2, 2)

    fig['layout'].update(showlegend=False, title='<b>Model performance report (5 folds)</b><br>'+subtitle,
                         autosize=False, height=830, width=830,
                         plot_bgcolor='black',
                         paper_bgcolor='black',
                         margin=dict(b=195), font=dict(color='white'))
    fig["layout"]["xaxis1"].update(color='white')
    fig["layout"]["yaxis1"].update(color='white')
    fig["layout"]["xaxis2"].update((dict(range=[0, 1], color='white')))
    fig["layout"]["yaxis2"].update(color='white')
    fig["layout"]["xaxis3"].update(
        dict(title="false positive rate"), color='white')
    fig["layout"]["yaxis3"].update(
        dict(title="true positive rate"), color='white')
    fig["layout"]["xaxis4"].update(dict(title="recall"), range=[
                                   0, 1.05], color='white')
    fig["layout"]["yaxis4"].update(dict(title="precision"), range=[
                                   0, 1.05], color='white')
    for i in fig['layout']['annotations']:
        i['font'] = titlefont = dict(color='white', size=14)
    py.plot(fig)


random_state = 42

# These are the parameters used in Grid Search

fit_params = {"early_stopping_rounds": 100,
              "eval_metric": 'auc',
              "eval_set": [(X, y)],
              'eval_names': ['valid'],
              'verbose': 0,
              'categorical_feature': 'auto'}

# These are the LightGBM model parameters ie. hyperparameters.
# Multiple values for each hyperparameter is passed and grid search basically tries all the combinations and finds out the best
# set of hyper parameters for the algorithm

param_test = {'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4],
              'n_estimators': [100, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000],
              'num_leaves': sp_randint(6, 50),
              'min_child_samples': sp_randint(100, 500),
              'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
              'subsample': sp_uniform(loc=0.2, scale=0.8),
              'max_depth': [-1, 1, 2, 3, 4, 5, 6, 7],
              'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
              'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
              'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

# number of combinations
n_iter = 300

# intialize lgbm and lunch the search
# we are using K fold Cross Validation and there are total of 5 folds that we are using

lgbm_clf = lgbm.LGBMClassifier(
    random_state=random_state, silent=True, metric='None', n_jobs=4)
grid_search = RandomizedSearchCV(
    estimator=lgbm_clf, param_distributions=param_test,
    n_iter=n_iter,
    scoring='accuracy',
    cv=5,
    refit=True,
    random_state=random_state,
    verbose=True)

# Our dataset is fitted on grid search to find best set of parameter values
grid_search.fit(X, y, **fit_params)

# opt_parameters contains the best parameter values for the algorithm for which the accuracy was the highest
opt_parameters = grid_search.best_params_

# We can then create a model with the optimum values for hyperparameters
lgbm_clf = lgbm.LGBMClassifier(**opt_parameters)

joblib.dump(lgbm_clf, 'NewLGBM.pkl')

model_performance(lgbm_clf, 'LightGBM')
scores_table(lgbm_clf, 'LightGBM')
