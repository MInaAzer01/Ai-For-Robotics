import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("insurance.csv")

df.head()

df.shape

df.info()

df.isnull().sum()

df.columns

df.describe()

df['sex'] = df['sex'].apply({'male':0, 'female':1}.get)
df['smoker'] = df['smoker'].apply({'yes':1, 'no':0}.get)
df['region'] = df['region'].apply({'southwest':1, 'southeast':2, 'northwest':3, 'northeast':4}.get)

df.head()
plt.figure(figsize=(10,7))
sns.heatmap(df.corr(), annot = True)
plt.show()

X = df.drop(['charges', 'sex'], axis=1)
y = df.charges

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train shpae: ", y_train.shape)
print("y_test shape: ", y_test.shape)

linreg = LinearRegression()

linreg.fit(X_train, y_train)
pred = linreg.predict(X_test)

from sklearn.metrics import r2_score

print("R2 score: ",(r2_score(y_test, pred)))

plt.scatter(y_test, pred)
plt.xlabel('Y test')
plt.ylabel('Y pred')
plt.show()

data = {'age':50, 'bmi':25, 'children':2, 'smoker':1, 'region':2}
index = [0]
cust_df = pd.DataFrame(data, index)
cust_df

cost_pred = linreg.predict(cust_df)
print("The medical insurance cost of the new customer is: ", cost_pred)