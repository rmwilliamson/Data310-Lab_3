# Lab 3

# Question 3
### Write your own code to import L3Data.csv into python as a data frame. Then save the feature values  'days online','views','contributions','answers' into a matrix x and consider 'Grade' values as the dependent variable. If you separate the data into Train & Test with test_size=0.25 and random_state = 1234. If we use the features of x to build a multiple linear regression model for predicting y then the root mean square error on the test data is close to:

```markdown
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from yellowbrick.regressor import ResidualsPlot

df = pd.read_csv('drive/MyDrive/Colab Notebooks/L3Data.csv')

y = df['Grade'].values
X = df.loc[ : , df.columns != ('questions' and 'Grade')].values

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1234)

# scale the data
Xs_train = scale.fit_transform(X_train)
Xs_test  = scale.transform(X_test)

from sklearn import linear_model
lm = linear_model.LinearRegression()

# fit the model and make prediction
lm.fit(Xs_train,y_train)
y_pred = lm.predict(Xs_test)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred)
math.sqrt(mse)
```
## Answer



# Question 7
### Write your own code to import L3Data.csv into python as a data frame. Then save the feature values  'days online','views','contributions','answers' into a matrix x and consider 'Grade' values as the dependent variable. If you separate the data into Train & Test with test_size=0.25 and random_state = 1234, then the number of observations we have in the Train data is

```markdown
len(X_train)
```

## Number of observations = 23
