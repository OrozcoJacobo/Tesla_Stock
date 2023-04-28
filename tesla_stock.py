#%%
#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import skillsnetwork
import sklearn
import scipy
import seaborn as sns

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from scipy.stats import norm
from scipy import stats


#%%
#Reading and understanding the data

#Loading the tesla stocks dataset
#Define the path to which you saved the excel data set of tesla stocks 
filepath = 'D:/IBM_Professional_Certificate/Course_2/data/tesla.csv'
#Import the data into pandas 
tesla_stocks = pd.read_csv(filepath)
#Lets observe the data to draw some initial insight
num_rows = 5
print('Let us observe the first five elements of our tesla stocks dataframe \n', tesla_stocks.head(num_rows))
print('\n')
print('Let us observe the last five elements of our tesla stocks dataframe \n', tesla_stocks.tail(num_rows))


#%%
#Find more information about the features and types 
tesla_stocks.info()


#%%
tesla_stocks['Close'].describe()


#%%
tesla_stocks_original = tesla_stocks.copy(deep = True)

tesla_stocks['Date'] = pd.to_datetime(tesla_stocks['Date'], dayfirst = True)
tesla_stocks.head(num_rows)


#%%
tesla_numerical_values = tesla_stocks.select_dtypes(include = ['float64', 'int64'])
tesla_numerical_values_corr = tesla_numerical_values.corr()['Close'][:]
top_features = tesla_numerical_values_corr[abs(tesla_numerical_values_corr) > 0.45].sort_values(ascending = False)
print('\nThere is {} strongly correlated values with Close:\n{}'.format(len(top_features), top_features))


#%%
tesla_numerical_values = tesla_stocks.select_dtypes(include = ['float64', 'int64'])
skew_limit = 0.5 # Define a limit above which we will log transform
skew_vals = tesla_numerical_values.skew()
skew_cols = (skew_vals
             .sort_values(ascending = False)
             .to_frame()
             .rename(columns = {0:'Skew'})
             .query('abs(Skew) > {}'.format(skew_limit)))
skew_cols


#%%
#Let's look at what happens to the volume feature when we apply log transformation visually.
volume_field = 'Volume'

#Create two subplots and a figure 
fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize = (10, 5))

#Create a histogram on the ax_before subplot
tesla_stocks[volume_field].hist(ax = ax_before)

#Apply a log transformation to this column
tesla_stocks[volume_field].apply(np.log).hist(ax = ax_after)

#Formating of titles etc
ax_before.set(title = 'Before np.log', ylabel = 'Frequency', xlabel = 'Price')
ax_after.set(title = 'After np.log', ylabel ='Frequency', xlabel = 'Price')
fig.suptitle('Field {}'.format(volume_field))

print('\nSkewness before np.log: \n', tesla_stocks[volume_field].skew())
print('\nSkewness after np.log: \n', tesla_stocks[volume_field].apply(np.log).skew())


#%%
#It works the volume data is now closer to a normal distribution, and more importantly it's within the margin I established

for col in skew_cols.index.values:
    if col == 'Close':
        continue
    tesla_stocks[col] = tesla_stocks[col].apply(np.log)

tesla_stocks.shape
print('\nSkewness after np.log: \n', tesla_stocks['Volume'].skew())


#%%
tesla_stocks_post_data_reg_copy = tesla_stocks.copy(deep = True)
tesla_stocks = tesla_stocks.drop('Adj Close', axis = 1)
tesla_stocks.head(num_rows)


#%%
sns.pairplot(tesla_stocks, plot_kws = dict(alpha = 1, edgecolor = 'none'))


#%%
#Split the data in train and test sets 
tesla_stocks_x = tesla_stocks.drop(['Close', 'Date'], axis = 1)
tesla_stocks_y = tesla_stocks.Close

print('tesla stocks features:\n', tesla_stocks_x.head())
print('\ntesla stocks objective:\n', tesla_stocks_y.head())

kf = KFold(shuffle = True, random_state = 72018, n_splits = 3)


#%%
scores_lr_NoStadard = []

lr = LinearRegression()

for train_index, test_index in kf.split(tesla_stocks_x):
    X_train, X_test, y_train, y_test = (tesla_stocks_x.iloc[train_index, :],
                                        tesla_stocks_x.iloc[test_index, :],
                                        tesla_stocks_y[train_index],
                                        tesla_stocks_y[test_index])
    
    lr.fit(X_train, y_train)
    
    y_pred = lr.predict(X_test)
    
    score = r2_score(y_test.values, y_pred)
    
    scores_lr_NoStadard.append(score)
    
scores_lr_NoStadard


#%%
tesla_stocks.head()

tesla_stocks_Date = tesla_stocks['Date']
tesla_stocks_LR = tesla_stocks.drop('Date',axis = 1)
print('Dates:\n', tesla_stocks_Date.head())
print('\nTesla stocks for LR:\n', tesla_stocks_LR.head())


#%%
#Split the data in train and test sets
tesla_stocks_train, tesla_stocks_test = train_test_split(tesla_stocks_LR, test_size = 0.3, random_state=42)

#Separate features from predictor
feature_cols = [x for x in tesla_stocks_train.columns if x != 'Close']
X_train = tesla_stocks_train[feature_cols]
y_train = tesla_stocks_train['Close']

X_test = tesla_stocks_test[feature_cols]
y_test = tesla_stocks_test['Close']


#%%
def rmse(ytrue, ypredicted):
    return np.sqrt(mean_squared_error(ytrue, ypredicted))


#%%
print('X_train:\n', X_train.head())
print('y_train:\n', y_train.head())

linearRegression = LinearRegression().fit(X_train, y_train)

prediction_lr = linearRegression.predict(X_test)

linearRegression_rmse = rmse(y_test, prediction_lr)
LinearRegression_r2 = r2_score(y_test, prediction_lr)

print('\nLinear Regression rmse: ', linearRegression_rmse)
print('\nLinear Regression r2: ', LinearRegression_r2)


#%%
f = plt.figure(figsize=(6,6))
ax = plt.axes()

ax.plot(y_test, linearRegression.predict(X_test), marker = 'o', ls='', ms = 3.0)

lim= (0, y_test.max())

ax.set(xlabel='Actual Close Price',
       ylabel='Predicted Close Price',
       xlim=lim,
       ylim=lim,
       title='Linear Regression Results')


#%%
alphas = [0.005, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 80]

ridgeCV = RidgeCV(alphas = alphas, cv = 3).fit(X_train, y_train)

prediction_ridgeCV = ridgeCV.predict(X_test)

ridgeCV_rmse = rmse(y_test, prediction_ridgeCV)
ridgeCV_r2 = r2_score(y_test, prediction_ridgeCV)

print('RidgeCV alpha: ', ridgeCV.alpha_, '\nRidgeCV rmse: ', ridgeCV_rmse, '\nRidgeCV r2 score: ', ridgeCV_r2)


#%%
alphas2 = np.array([1e-5, 5e-5, 0.0001, 0.0005, 0.0007, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5])

lassoCV = LassoCV(alphas = alphas2, max_iter = 1000000000, cv = 3).fit(X_train, y_train)

prediction_LassoCV = lassoCV.predict(X_test)

lassoCV_rmse = rmse(y_test, prediction_LassoCV)

lassoCV_r2 = r2_score(y_test, prediction_LassoCV)

print('LassoCV alpha: ', lassoCV.alpha_, '\nLassoCV rmse: ', lassoCV_rmse, '\nLassoCV r2 score: ', lassoCV_r2)


#%%
l1_ratios = np.linspace(0.1, 0.9, 9)

elasticNetCV = ElasticNetCV(alphas = alphas2, l1_ratio=l1_ratios, max_iter=1000000000).fit(X_train , y_train)

prediction_ElasticNetCV = elasticNetCV.predict(X_test)

elasticNetCV_rmse = rmse(y_test, prediction_ElasticNetCV)

elasticNetCV_r2 = r2_score(y_test, prediction_ElasticNetCV)


print('ElasticNetCV alpha: ', elasticNetCV.alpha_, '\nElasticNetCV rmse: ', elasticNetCV_rmse, '\nElasticNetCV r2 score: ', elasticNetCV_r2)


#%%
rmse_vals = [linearRegression_rmse, ridgeCV_rmse, lassoCV_rmse, elasticNetCV_rmse]
r2_scores = [LinearRegression_r2, ridgeCV_r2, lassoCV_r2, elasticNetCV_r2]

data = {'RMSE' : rmse_vals, 'R2 Scores': r2_scores}

labels = ['Linear', 'Ridge', 'Lasso', 'ElasticNet']

comparison_dataframe = pd.DataFrame(data = data, index = labels)
comparison_dataframe



#%%
#Lets start by adding some polynomial features and scaling our data

pf = PolynomialFeatures(degree = 3)
ss = StandardScaler()

scores_pf = []
alphas_pf = np.geomspace(0.6, 6.0, 20)
for alpha in alphas_pf:
    lasso_pf = Lasso(alpha = alpha, max_iter = 1000000)
    
    estimator = Pipeline([
        ('scaler', ss),
        ('make_higher_degree', pf),
        ('lassp_regression', lasso_pf)])
    
    predictions = cross_val_predict(estimator, tesla_stocks_x, tesla_stocks_y, cv = kf)
    
    pf_lasso_r2_score = r2_score(tesla_stocks_y, predictions)
    
    scores_pf.append(score)

scores_pf


#%%
plt.semilogx(alphas_pf, scores_pf)


#%%