from matplotlib import colors
import pandas as pd
from seaborn.categorical import barplot
from seaborn.palettes import color_palette
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import datetime 

#decalre train and test data
train_data = pd.read_csv('source/train.csv')
test_data = pd.read_csv('source/test.csv')
sample_sub = pd.read_csv('source/sample_submission.csv')

df = pd.concat([train_data,test_data],sort=False)

#split data as year
df['date'] = pd.to_datetime(df['date'])

years = [df.date.dt.year]

train_data_2013 = df[df.date.dt.year == 2013]
train_data_2014 = df[df.date.dt.year == 2014]
train_data_2014 = df[df.date.dt.year == 2014]
train_data_2015 = df[df.date.dt.year == 2015]
train_data_2016 = df[df.date.dt.year == 2016]
train_data_2017 = df[df.date.dt.year == 2017]
train_data_2018 = df[df.date.dt.year == 2018]


# train_data_2014['month'] = train_data_2014.date.dt.month
# store_index_2014 = train_data_2014.set_index('store')

#sales detailes from 2013
sales_2013 = pd.DataFrame({
    'item_count': train_data_2013.groupby('date')['item'].count(),
    'item_sum':train_data_2013.groupby('date')['item'].sum(),
    'sales_avg':train_data_2013.groupby('date')['sales'].mean(),
    'sales_sum':train_data_2013.groupby('date')['sales'].sum()
})

#sales detailes from 2014
sales_2014 = pd.DataFrame({
    'item_count': train_data_2014.groupby('date')['item'].count(),
    'item_sum':train_data_2014.groupby('date')['item'].sum(),
    'sales_avg':train_data_2014.groupby('date')['sales'].mean(),
    'sales_sum':train_data_2014.groupby('date')['sales'].sum()
})

#sales detailes from 2015
sales_2015 = pd.DataFrame({
    'item_count': train_data_2015.groupby('date')['item'].count(),
    'item_sum':train_data_2015.groupby('date')['item'].sum(),
    'sales_avg':train_data_2015.groupby('date')['sales'].mean(),
    'sales_sum':train_data_2015.groupby('date')['sales'].sum()
})

#sales detailes from 2016
sales_2016 = pd.DataFrame({
    'item_count': train_data_2016.groupby('date')['item'].count(),
    'item_sum':train_data_2016.groupby('date')['item'].sum(),
    'sales_avg':train_data_2016.groupby('date')['sales'].mean(),
    'sales_sum':train_data_2016.groupby('date')['sales'].sum()
})

#sales detailes from 2017
sales_2017 = pd.DataFrame({
    'item_count': train_data_2017.groupby('date')['item'].count(),
    'item_sum':train_data_2017.groupby('date')['item'].sum(),
    'sales_avg':train_data_2017.groupby('date')['sales'].mean(),
    'sales_sum':train_data_2017.groupby('date')['sales'].sum()
})

#sales detailes from 2018
sales_2018 = pd.DataFrame({
    'item_count': train_data_2018.groupby('date')['item'].unique().count(),
    'item_sum':train_data_2018.groupby('date')['item'].sum(),
    'sales_avg':train_data_2018.groupby('date')['sales'].mean(),
    'sales_sum':train_data_2018.groupby('date')['sales'].sum()
})

#print the year sales average
# fig,axis = plt.subplots(nrows=6,constrained_layout=True)
# axis[0].plot(sales_2013.index,sales_2013['sales_avg'])
# axis[1].plot(sales_2014.index,sales_2014['sales_avg'])
# axis[2].plot(sales_2015.index,sales_2015['sales_avg'])
# axis[3].plot(sales_2016.index,sales_2016['sales_avg'])
# axis[4].plot(sales_2017.index,sales_2017['sales_avg'])
# axis[5].plot(sales_2018.index,sales_2018['sales_avg'])

# plt.savefig('years_sales_avg.png')

# plt.show()

sales_month_2013 = pd.DataFrame({
    'item_count':train_data_2013.groupby(train_data_2013.date.dt.month)['item'].unique().count(),
    'item_sum':train_data_2013.groupby(train_data_2013.date.dt.month)['item'].sum(),
    'sales_avg':train_data_2013.groupby(train_data_2013.date.dt.month)['sales'].mean(),
    'sales_sum':train_data_2013.groupby(train_data_2013.date.dt.month)['sales'].sum(),
})

sales_month_2014 = pd.DataFrame({
    'item_count':train_data_2014.groupby(train_data_2014.date.dt.month)['item'].unique().count(),
    'item_sum':train_data_2014.groupby(train_data_2014.date.dt.month)['item'].sum(),
    'sales_avg':train_data_2014.groupby(train_data_2014.date.dt.month)['sales'].mean(),
    'sales_sum':train_data_2014.groupby(train_data_2014.date.dt.month)['sales'].sum(),
})

sales_month_2015 = pd.DataFrame({
    'item_count':train_data_2015.groupby(train_data_2015.date.dt.month)['item'].unique().count(),
    'item_sum':train_data_2015.groupby(train_data_2015.date.dt.month)['item'].sum(),
    'sales_avg':train_data_2015.groupby(train_data_2015.date.dt.month)['sales'].mean(),
    'sales_sum':train_data_2015.groupby(train_data_2015.date.dt.month)['sales'].sum(),
})

sales_month_2016 = pd.DataFrame({
    'item_count':train_data_2016.groupby(train_data_2016.date.dt.month)['item'].unique().count(),
    'item_sum':train_data_2016.groupby(train_data_2016.date.dt.month)['item'].sum(),
    'sales_avg':train_data_2016.groupby(train_data_2016.date.dt.month)['sales'].mean(),
    'sales_sum':train_data_2016.groupby(train_data_2016.date.dt.month)['sales'].sum(),
})

sales_month_2017 = pd.DataFrame({
    'item_count':train_data_2017.groupby(train_data_2017.date.dt.month)['item'].unique().count(),
    'item_sum':train_data_2017.groupby(train_data_2017.date.dt.month)['item'].sum(),
    'sales_avg':train_data_2017.groupby(train_data_2017.date.dt.month)['sales'].mean(),
    'sales_sum':train_data_2017.groupby(train_data_2017.date.dt.month)['sales'].sum(),
})

sales_month_2018 = pd.DataFrame({
    'item_count':train_data_2018.groupby(train_data_2018.date.dt.month)['item'].unique().count(),
    'item_sum':train_data_2018.groupby(train_data_2018.date.dt.month)['item'].sum(),
    'sales_avg':train_data_2018.groupby(train_data_2018.date.dt.month)['sales'].mean(),
    'sales_sum':train_data_2018.groupby(train_data_2018.date.dt.month)['sales'].sum(),
})

# fig,ax = plt.subplots(2,3,figsize=(15,5),sharey=True)
# sns.barplot(ax=ax[0][0],x=sales_month_2013.index,y=sales_month_2013['sales_avg'])
# sns.barplot(ax=ax[0][1],x=sales_month_2014.(index,y=sales_month_2014['sales_avg'])
# sns.barplot(ax=ax[0][2],x=sales_month_2015.index,y=sales_month_2015['sales_avg'])
# sns.barplot(ax=ax[1][0],x=sales_month_2016.index,y=sales_month_2016['sales_avg'])
# sns.barplot(ax=ax[1][1],x=sales_month_2017.index,y=sales_month_2017['sales_avg'])
# sns.barplot(ax=ax[1][2],x=sales_month_2018.index,y=sales_month_2018['sales_avg'])
# fig.suptitle('Yıllık stok veri analizi')
# plt.show()

#plot avg sales by month for each year
# start = 3
# finish = 7
# fig,ax = plt.subplots(2,3,figsize=(15,5),sharey=True)
# for i in range(start,finish+1):
#     row_id = int((i-start)/3)
#     column_id = i%3
#     sns.barplot(ax=ax[row_id][column_id],x=sales_month_2013.index,y=sales_month_2013['sales_avg'])
#     fig.suptitle('Yıllık stok veri analizi')
# plt.show()

from sklearn.naive_bayes import *
from sklearn.linear_model import LinearRegression
import numpy as np

# store_1 = df[df['store'] == 1] #create data frame for store 1

# store_1 = store_1.set_index(store_1.date.dt.year)
# store_1 = store_1.drop('id',axis=1) #drop id column
# store_1 = store_1['sales']
# train = store_1.iloc[:-4865]
# test = store_1.iloc[-4865:]

# trend_removed = train.diff()

# print(trend_removed)
# plt.figure(figsize=(12,6))
# plt.plot(trend_removed)
# plt.show()

#function to parse date 
def date_parser(data_set):
    return datetime.datetime.strptime(data_set,'%Y-%m-%d')

plt.figure(figsize=(12,6))
sales_all_month = pd.read_csv('source/sales_all_month.csv',parse_dates=[0],index_col=0,squeeze=True,date_parser=date_parser)
sales_all_month = sales_all_month.drop('sum',axis=1)


sales_all_month = sales_all_month.asfreq(pd.infer_freq(sales_all_month.index))
# start_date = datetime.datetime(2013,1,1)
# end_date = datetime.datetime(2017,12,31)

# sales_all_month_modified = sales_all_month[start_date:end_date]

# t_train_sales = np.arange(len(sales_all_month)).reshape(-1,1)
diff_sales = sales_all_month.diff()
# mask = (diff_sales.index> datetime.datetime(2013,1,1)) & (diff_sales.index<datetime.datetime(2017,12,31))

# t_train = np.arange(len(sales_all_month)).reshape(-1,1)

# diff_sales = diff_sales / ((t_train+1)**(1/2)).reshape(-1)


# diff_sales = diff_sales/((t_train_sales+1)**(1
# bins = bins[:,0]/2)).reshape(-1)

# plt.plot(diff_sales)
# plt.show()

'''
####### bining data #############
'''

diff_sales = diff_sales.iloc[1:]

n_bins = len(diff_sales)
bins = np.linspace(diff_sales.min(),diff_sales.max(),n_bins)
bins = bins[:,0]
binned = np.digitize(diff_sales,bins)
binned = list(binned)
binned_series = pd.Series(binned,index = diff_sales.index)

bin_means = {}

for biin in range(1,n_bins):
    bin_means[biin] = diff_sales.iloc[[biin]]['avg']

lagged_list = []

for s in range(13):
    lagged_list.append(binned_series.shift(s))

lagged_frame = pd.concat(lagged_list,1).dropna()

train_X = lagged_frame.iloc[:,1:]
train_Y = lagged_frame.iloc[:,0]
train_Y = train_Y.astype('int')

###### create module #####

### functio to return mean prediction

def get_mean(pred):
    # print(bin_means[pred[0]][0])
    return bin_means[pred[0]][0]

model = GaussianNB()

model_fit = model.fit(train_X,train_Y)

pred_insample = model.predict(train_X)

pred_insample = pd.DataFrame(pred_insample,index=train_Y.index)


resulting_preiction = pd.Series(np.nan,index=train_Y.index)
print("----------------------------------")
for i in range(len(pred_insample)+1):
    resulting_preiction.iloc[i] = get_mean(pred_insample.values[i])

# plt.figure(figsize=(12,6))
# plt.plot(diff_sales)
# plt.plot(resulting_preiction)
# plt.show()

# sales_all_month.set_index([sales_all_month.index.date.dt.year,sales_all_month.index.date.dt.month])
##########################################

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

# plot_acf(sales_2013['sales_avg'],lags=50)
# plt.show()

# print(train_data_2013.groupby(['store','date']).keys('store'))

# plt.plot(train_data_2013.groupby('date').index,train_data_2013.groupby(['store','date'])['sales'].mean())
# plt.show()

'''
# data = pd.read_csv('source/sample_submission.csv')
data = pd.read_csv('source/train.csv')

header_data = data.head()

#start statiscal operations

test_set = data[:5000] #assign dataframe

print(data.describe())
print('null values')
print(data.isnull().sum())
print(data.groupby(['date']).head())
#ploting the IQR

plot_data = pd.DataFrame({
    'date':data['date'],
    'sales':data['sales']
})

# plot_data.to_csv('plot.csv',index=False)

plt.figure(figsize=(10,10))

# data.cumsum()
data.plot()

#sns.boxplot(data['sales'])

# sns.countplot(x='item',y='sales',data=data['item'].groupby(data['sales']))

# sns.barplot(x='date',y='sales',data=plot_data)

# plt.show()

'''

#print(test_data.describe())  #show statisticsal info
#print(train_data.describe()) #show statisticsal info

#drop date column to easy calssify
'''
train_data = train_data[:45000].drop(['date'],axis=1)
test_data = test_data.drop(['date'],axis=1)

X_train,X_test,Y_train,Y_test = train_test_split(train_data,test_data,random_state=0)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
# Decision Tree Regression 
clf = DecisionTreeRegressor(max_depth=2,random_state=0)

clf.fit(X_train,Y_train)

score = clf.score(X_test,Y_test)

print(f'Accuracy of tree deceision regression on training is {clf.score(X_train,Y_train)}')
print(f'Accuracy of tree deceision regression on test is {clf.score(X_test,Y_test)}')

#plot regression 
'''


'''
plt.figure()
plt.scatter(train_data,test_data,edgecolors='black',c='darkorange',label='data')
plt.scatter(X_test,clf.predict(X_test),linewidths=2)
plt.xlabel("Data")
plt.ylabel("Decision Tree Regression")
plt.legend()
plt.savefig('decision_tree_regression.png')
plt.show()
'''


'''
# using KNN
knnClf = KNeighborsClassifier(n_neighbors=3)
knnClf = knnClf.fit(X_train,Y_train['item'])
plt.figure()
sns.heatmap(test_data.corr(),annot=True,fmt='.2f')
plt.show()

print(f'Accuracy of KNN on training is {knnClf.score(X_train,Y_train["item"])}')
print(f'Accuracy of KNN on test is {knnClf.score(X_test,Y_test["item"])}')

'''
    