
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv("/content/restaurant_data.csv")
df
df.head()
df.info()
df.fillna(method='backfill',inplace=True)
valid_cuisines=df['cuisine'].value_counts()[df['cuisine'].value_counts()>=2].index
df=df[df['cuisine'].isin(df['cuisine'].value_counts()[df['cuisine'].value_counts()>=2].index)].copy()
df['cuisine']=LabelEncoder().fit_transform(df['cuisine'])
df=pd.get_dummies(df,columns=['location','type'],drop_first=True)
X=df.drop(['cuisine','restaurant_name'],axis=1)
y=df['cuisine']
for col in X.columns:
    if X[col].dtype=='object':
        print(f"column '{col}' is still of type of object")
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)
class_weight=compute_class_weight(class_weight='balanced',classes=np.unique(y_train),y=y_train)
class_weights_dict=dict(zip(np.unique(y_train),compute_class_weight(class_weight='balanced',classes=np.unique(y_train),y=y_train)))
X_train_scaled=StandardScaler().fit_transform(X_train)
X_test_scaled=StandardScaler().fit_transform(X_test)
LogisticRegression(max_iter=1000,class_weight='balanced').fit(X_train_scaled,y_train)
lr_pred=LogisticRegression(max_iter=1000,class_weight='balanced').fit(X_train_scaled,y_train).predict(X_test_scaled)
RandomForestClassifier(class_weight=class_weights_dict,random_state=42).fit(X_train,y_train)
rf_pred=RandomForestClassifier(class_weight=class_weights_dict,random_state=42).fit(X_train,y_train).predict(X_test)
XGBClassifier(use_label_encoder=False,eval_metric="mlogloss").fit(X_train,y_train)
xgb_pre=XGBClassifier(use_label_encoder=False,eval_metric="mlogloss").fit(X_train,y_train).predict(X_test)
def evaluate_model(y_True,y_pred,model_name):
    print(f"\n{model_name}Evaluated")
    print("Accuracy",accuracy_score(y_True,y_pred))
    print("Classification Report:\n",classification_report(y_True,y_pred))
evaluate_model(y_test,lr_pred,"Logistic Regression")
evaluate_model(y_test,lr_pred,"Random Forest")
evaluate_model(y_test,xgb_pre,"XGboost")
def plot_matrix(y_True,y_pred,model_name):
    plt.figure(figsize=(16,9))
    sns.heatmap(confusion_matrix(y_True,y_pred),annot=True,fmt='d',cmap="plasma")
    plt.xlabel("predicate")
    plt.ylabel("Actual")
    plt.show()
plot_matrix(y_test,rf_pred,"Random Forest")
importance = RandomForestClassifier(class_weight=class_weights_dict, random_state=42).fit(X_train, y_train).feature_importances_
feature=X.columns
importance_df = pd.DataFrame({'Feature': feature, 'Importance': importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(16,9))
sns.barplot(data=importance_df.head(10), x='Importance', y='Feature')
plt.title('Important Features (Random Forest)')
plt.show()
param={
    "n_estimators":[100,200],
    "max_depth":[10,20,None],
    "min_samples_split":[2,5,10],
    "min_samples_leaf":[1,2,4]
}
grid_s = GridSearchCV(RandomForestClassifier(class_weight=class_weights_dict), param, cv=3, scoring='accuracy')
grid_s.fit(X_train, y_train)

best_rf = grid_s.best_estimator_
best_rf_pred = best_rf.predict(X_test)

evaluate_model(y_test, best_rf_pred, "Tuned Random Forest")