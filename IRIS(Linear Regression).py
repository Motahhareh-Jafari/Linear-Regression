import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

data= sns.load_dataset('iris')
print(data)

x=data[['sepal_width']]
y= data[['petal_width']]

x_train, x_test,y_train, y_test= train_test_split(x, y,test_size=0.3, random_state=42)

model= LinearRegression()
model.fit(x_train, y_train)

print('zarayeb:', model.coef_)
print('sabet', model.intercept_)

y_pred = model.predict(x_test)
print('Mean Squared Error', mean_squared_error(y_test, y_pred))
print('R_squared:', r2_score(y_test, y_pred))



line_x = np.linspace(x['sepal_width'].min(), x['sepal_width'].max(), 100)
line_y = model.predict(pd.DataFrame({'sepal_width': line_x}))[:, list(y.columns).index('petal_width')]

plt.plot(line_x, line_y, color='red', label='Regression Line')
plt.xlabel('sepal_width')
plt.ylabel('petal_width')
plt.title(f'Scatter Plot of sepal_width vs petal_width')
plt.legend()
plt.show()