from numpy.random.mtrand import triangular
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor

train_data = pd.read_csv('source/train.csv')
test_data = pd.read_csv('source/test.csv')
sample_sub = pd.read_csv('source/sample_submission.csv')


df = pd.concat([train_data,test_data],sort=False)

print(test_data.shape)

# df['date'] = pd.to_datetime(df['date'])

# df = df.drop('id',axis=1)
# sns.barplot(x=df.date.dt.month,y=df['sales'],data=df)
# plt.show()
'''
train_data = pd.read_csv('source/train.csv')
test_data = pd.read_csv('source/test.csv')

train_data = train_data[:45000].drop(['date'],axis=1)
test_data = test_data.drop(['date'],axis=1)

X_train,X_test,Y_train,Y_test = train_test_split(train_data,test_data,random_state=0)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
clf = DecisionTreeRegressor()
clf = clf.fit(X_train,Y_train)

print(f'Accuracy of tree deceision regression on training is {clf.score(X_train,Y_train)}')
print(f'Accuracy of tree deceision regression on test is {clf.score(X_test,Y_test)}')


# using KNN
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=3)
clf = clf.fit(X_train,Y_train['item'])

print(f'Accuracy of tree deceision regression on training is {clf.score(X_train,Y_train["item"])}')
print(f'Accuracy of tree deceision regression on test is {clf.score(X_test,Y_test["item"])}')
'''