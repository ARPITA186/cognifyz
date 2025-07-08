
import pandas as pd
import folium
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
df=pd.read_csv('/content/restaurant_id.csv')
map=[df['latitude'].mean(),df['longitude'].mean()]
restaurant=folium.Map(location=map,zoom_start=5)
for idx,row in df.iterrows():
    folium.CircleMarker(
        location=[row['latitude'],row['longitude']],
        radius=5,
        color='red',
        fill=True,
        fill_opacity=0.6,
        popup=f"{row['restaurant_name']}({row['city']})"
    ).add_to(restaurant)
display(restaurant)
restaurant.save('map.jpg')
#restaurant.show_in_browser()
city=df.groupby('city').size().reset_index(name='restaurant').sort_values(by='restaurant',ascending=False)
print("Total count of Restaurent per city",city)
city_rate=df.groupby('city')['rating'].mean().reset_index().sort_values(by='rating',ascending=False)
print("Mean of rating per city based",city_rate)
plt.figure(figsize=(16,9))
sns.barplot(data=city,x='city',y='restaurant')
plt.xlabel('All the Cities')
plt.ylabel("Total Restaurant")
plt.title("City wise restaurant numbers")
plt.grid()
plt.xticks(rotation=90)
plt.show()
plt.figsize=(16,9)
sns.barplot(data=city_rate,x='city',y='rating')
plt.title("Mean Restaurant ratings per city")
plt.xlabel("All cities")
plt.ylabel("Mean Rating")
plt.xticks(rotation=90)
plt.grid()
plt.show()