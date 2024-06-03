#Importing data
import pandas as pd

df = pd.read_csv("seattle-weather.csv")
df.head()
df.describe()
print(df.describe())
print("Data set shape: ", df.shape)
print(df)

df.isnull().sum()
print(df.isnull().sum())

def labelencoding(c):
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    df[c] = le.fit_transform(df[c])
    df[c].unique()
labelencoding("weather")
cols = ['precipitation' , 'temp_max', 'temp_min', 'wind']
###Normalize the columns
def normalize(df, cols):
    for x in cols:
        df[x] = df[x] / df[x].max()
    normalize(df, cols)
print("Data set normalized")
print(df)
df = df.drop('date',axis=1)
df

x = df.drop('weather',axis=1)
y = df['weather']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#importing the xgbooter and applying the classifier to the odel
from xgboost import XGBClassifier
xgmodel = XGBClassifier()
xgmodel.fit(X_train, y_train)
##xgmodel.get_params()
print(xgmodel)

print(xgmodel.score(X_test, y_test))

y_hat=xgmodel.predict(X_test)
print(y_hat)

print('\n\ty_test Results\b\n')
print(y_test)
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
print("accuracy score: ", accuracy_score(y_test, y_hat))
print("Classification report: ", classification_report(y_test, y_hat))


