
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df=pd.read_csv('/content/Dataset .csv')
df.head()
df.fillna(method='backfill',inplace=True)
cat=df.select_dtypes(include=['object']).columns.tolist()
label_encoder = {}
for c in cat:
    df[c]=LabelEncoder().fit_transform(df[c])
    label_encoder[c] = LabelEncoder()
x= df.drop("Aggregate rating",axis=1)
y=df["Aggregate rating"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=45)
print("<<<<<<<<<<<<<<<<<<Linear Regression Model>>>>>>>>>>>>>>>>>>")
LinearRegression().fit(x_train,y_train)
r2=r2_score(y_test,LinearRegression().fit(x_train, y_train).predict(x_test))
mse = mean_squared_error(y_test, LinearRegression().fit(x_train, y_train).predict(x_test))
print(f"Linear Regression MSE:{mse:2f}")
print(f"Linear Regression R2 Score:{r2:2f}")
coefficient=pd.Series(LinearRegression().fit(x_train,y_train).coef_,index=x.columns)
print("\n Top positive possitive Influencing factors are:")
print(coefficient.sort_values(ascending=False).head(5))
print("\n Top negative Influencing factors are:")
print(coefficient.sort_values().head(5))
print("\n <<<<<<<<<<<<<<<<<<Decision Tree Regression Model>>>>>>>>>>>>>>>>>>")
DecisionTreeRegressor(random_state=45).fit(x_train,y_train)
mse1=mean_squared_error(y_test,DecisionTreeRegressor(random_state=45).fit(x_train,y_train).predict(x_test))
r3=r2_score(y_test,DecisionTreeRegressor(random_state=45).fit(x_train,y_train).predict(x_test))
print(f"Decision Tree MSE:{mse1:2f}")
print(f"Decision Tree R² Score:{r3:2f}")
feature=pd.Series(DecisionTreeRegressor(random_state=45).fit(x_train,y_train).feature_importances_,index=x.columns)
top_ft=feature.sort_values(ascending=False).head(5).reset_index()
top_ft.columns = ['Feature', 'Importance']
data=pd.DataFrame({
    'Feature':np.repeat(top_ft['Feature'].values,10),
    'Importance':np.concatenate([np.random.normal(loc=imp,scale=0.01,size=10)
                 for imp in top_ft['Importance'].values])
})
plt.figure(figsize=(16,9))
sns.boxplot(data=data,x='Importance',y='Feature',palette='Set2')
plt.grid(True)
plt.title("Boxplot of Feature Importance",fontsize=30,fontstyle='italic',color='brown')
plt.xlabel("Importance",fontsize=30,color='brown',fontstyle='oblique')
plt.ylabel("Feature",fontsize=30,color='brown',fontstyle='oblique')
plt.savefig('Feature_Importance.png')
plt.show()