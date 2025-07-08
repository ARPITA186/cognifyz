
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
df=pd.read_csv("/content/hjh.csv")

df.fillna({"Cuisine":"Not Defined","Cost":"In Range","Rating":df["Rating"].mean()},inplace=True)

df1=pd.concat([df[['Name','Rating']],pd.get_dummies(df[['Cuisine','Cost']])],axis=1)

print("Avialable Cuisine:",df['Cuisine'].unique())
cui=input("Enter your preffered cuisine").strip()
print("Avialable price: Low, Medium, High")
cos=input("Enter your preffered price").strip().capitalize()
user_input={'Cuisine':cui,'Cost':cos}
similarity=cosine_similarity(pd.get_dummies(pd.DataFrame([user_input])).reindex(columns=pd.get_dummies(df[['Cuisine', 'Cost']]).columns,fill_value=0),pd.get_dummies(df[['Cuisine', 'Cost']]))

df['Similarity_Score']=similarity[0]
recomend=df.sort_values(by='Similarity_Score',ascending=False).head()
print("\n Top Recommended Restaurants:")
print(recomend[['Name','Cuisine','Cost','Rating']])