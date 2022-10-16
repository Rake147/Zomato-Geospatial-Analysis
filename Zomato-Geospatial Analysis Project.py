#!/usr/bin/env python
# coding: utf-8

# In[102]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[103]:


df=pd.read_csv('C:/Users/Rakesh/3-Zomato Data Analysis/zomato.csv')


# In[104]:


df.head()


# In[105]:


df.columns


# In[106]:


df.dtypes


# In[107]:


df.shape


# In[108]:


df.isnull().sum()


# In[109]:


feature_na=[feature for feature in df.columns if df[feature].isnull().sum()>0]


# In[110]:


feature_na


# In[111]:


for feature in feature_na:
    print(' {} has {} missing values'.format(feature,np.round(df[feature].isnull().sum()/len(df)*100,4)))


# In[112]:


df['rate'].unique()


# In[113]:


df.dropna(axis='index',subset=['rate'],inplace=True)


# In[114]:


df.shape


# In[115]:


def split(x):
    return x.split('/')[0]


# In[116]:


df['rate']=df['rate'].apply(split)


# In[117]:


df.head()


# In[118]:


df.replace('NEW',0,inplace=True)


# In[119]:


df.replace('-',0,inplace=True)


# In[120]:


df['rate'].dtypes


# In[121]:


df['rate']=df['rate'].astype(float)


# In[122]:


df.head()


# # Calculate average rating of each restaurant

# In[123]:


df_rate=df.groupby('name')['rate'].mean().to_frame().reset_index()


# In[124]:


df_rate


# In[125]:


df_rate.columns=['Restaurant','Avg_rating']


# In[126]:


df_rate.head()


# In[127]:


import seaborn as sns


# In[128]:


sns.distplot(df_rate['Avg_rating'])


# # Top Restaurant chains in Bengaluru

# In[129]:


chains=df['name'].value_counts()[0:20]
sns.barplot(x=chains,y=chains.index)
plt.title('Most Famous restaurants in Bengaluru')
plt.xlabel('Number of Outlets')


# # How many of the restaurants do not accept Online Orders

# In[130]:


x=df['online_order'].value_counts()
x


# In[131]:


import plotly.express as px


# In[132]:


labels=['Accepted','Not Accepted']


# In[133]:


px.pie(df,values=x,labels=labels,title="Pie Chart")


# In[134]:


y=df['book_table'].value_counts()
y


# In[135]:


import plotly.graph_objs as go
from plotly.offline import iplot


# In[136]:


labels=['Not booked', 'Booked']


# In[137]:


trace=go.Pie(labels=labels,values=y,hoverinfo='label+percent',textinfo='value')
iplot([trace])


# # In depth Analysis of Types of Restaurants we have

# In[138]:


df['rest_type'].isnull().sum()


# In[139]:


df['rest_type'].dropna(inplace=True)


# In[140]:


df['rest_type'].isna().sum()


# In[141]:


df.groupby('name')['votes'].sum().nlargest(20).plot.bar()


# In[142]:


trace1=go.Bar(x=df.groupby('name')['votes'].sum().nlargest(20).index,
              y=df.groupby('name')['votes'].sum().nlargest(20))


# In[143]:


iplot([trace1])


# # Total Restaurants at different locations of Bengaluru

# In[144]:


restaurant=[]
location=[]
for key,location_df in df.groupby('location'):
    location.append(key)
    restaurant.append(len(location_df['name'].unique()))


# In[145]:


df_total=pd.DataFrame(zip(location,restaurant))
df_total.head()


# In[146]:


df_total.columns=['location','restaurant']


# In[147]:


df_total.head()


# In[148]:


df_total.sort_values(by='restaurant').tail(10).plot.bar()


# In[149]:


cuisines=df['cuisines'].value_counts()[0:10]
cuisines


# In[150]:


trace3=go.Bar(x=df['cuisines'].value_counts()[0:10].index, y=df['cuisines'].value_counts()[0:10])


# In[151]:


iplot([trace3])


# # Analyse aproximate cost for two peoples

# In[152]:


df.columns


# In[153]:


df['approx_cost(for two people)'].isnull().sum()


# In[154]:


df.dropna(axis='index', subset=['approx_cost(for two people)'],inplace=True)


# In[155]:


df['approx_cost(for two people)'].isnull().sum()


# In[156]:


df['approx_cost(for two people)'].dtypes


# In[157]:


df['approx_cost(for two people)'].unique()


# In[158]:


df['approx_cost(for two people)']=df['approx_cost(for two people)'].apply(lambda x: x.replace(',',''))


# In[163]:


df['approx_cost(for two people)'].unique()


# In[164]:


df['approx_cost(for two people)']=df['approx_cost(for two people)'].astype(int)


# In[165]:


df['approx_cost(for two people)'].dtypes


# In[166]:


sns.distplot(df['approx_cost(for two people)'])


# # Analyse approx cost vs rating 

# In[167]:


sns.scatterplot(x='rate',y='approx_cost(for two people)',data=df)


# In[168]:


sns.scatterplot(x='rate',y='approx_cost(for two people)',hue='online_order',data=df)


# # Votes for restaurant accepting vs not accepting online orders

# In[169]:


sns.boxplot(x='online_order', y='votes', data=df)


# In[170]:


px.box(df,x='online_order',y='votes')


# In[171]:


px.box(df,x='online_order',y='approx_cost(for two people)')


# # Luxurious restaurant in Bengaluru

# In[172]:


df['approx_cost(for two people)'].min()


# In[173]:


df['approx_cost(for two people)'].max()


# In[174]:


df[df['approx_cost(for two people)']==6000]['name']


# # Top 10 restaurants with approx cost for two people

# In[176]:


df_res=df.groupby('approx_cost(for two people)')['name'].unique().to_frame()


# In[177]:


df_res.head()


# In[178]:


df_res=df.groupby('approx_cost(for two people)')['name'].unique().to_frame().reset_index()


# In[179]:


df_res.head()


# In[182]:


df_res.tail(10)


# In[183]:


df_res.head(10)


# # Restaurants below 500 

# In[184]:


df_res[df_res['approx_cost(for two people)']<=500]['name'] 


# In[186]:


df_res[df_res['approx_cost(for two people)']<=500]


# # Restaurants that have good rating >4 and that are coming under budget 

# In[187]:


df[(df['rate']>4) & (df_res['approx_cost(for two people)']<=500)].shape


# In[188]:


len(df[(df['rate']>4) & (df_res['approx_cost(for two people)']<=500)]['name'].unique())


# In[189]:


df_res.shape


# In[190]:


data=df.copy()


# In[191]:


data.set_index('name',inplace=True)


# In[192]:


data.head()


# In[193]:


data['approx_cost(for two people)'].nlargest(10).plot.bar()


# In[194]:


df[(df['rate']>4) & (df['approx_cost(for two people)']<=500)].shape


# In[195]:


len(df[(df['rate']>4) & (df['approx_cost(for two people)']<=500)]['name'].unique())


# # Total various affordable hotels at all locations of Bengaluru

# In[196]:


df_new=df[(df['rate']>4) & (df['approx_cost(for two people)']<=500)]


# In[197]:


df_new.head()


# # Total various affordable hotels at all locations of bengaluru

# In[198]:


location=[]
total=[]
for loc, location_df in df_new.groupby('location'):
    location.append(loc)
    total.append(len(location_df['name'].unique()))


# In[199]:


location_df=pd.DataFrame(zip(location,total))
location_df.head()


# In[200]:


location_df.columns=['location','restaurant']


# In[201]:


location_df.head()


# # Best Budget restaurants in any location

# In[202]:


def return_budget(location,restaurant):
    budget=df[(df['approx_cost(for two people)']<=400)&(df['location']==location)&(df['rate']>4)&(df['rest_type']==restaurant)]
    return (budget['name'].unique())


# In[203]:


return_budget('BTM','Quick Bites')


# In[204]:


restaurant_location=df['location'].value_counts()[0:20]


# In[205]:


sns.barplot(restaurant_location,restaurant_location.index)


# # Finding the latitudes and longitudes for each of the location

# In[206]:


locations=pd.DataFrame({'Name':df['location'].unique()})


# In[207]:


locations.head()


# In[208]:


get_ipython().system('pip install geopy')


# In[209]:


from geopy.geocoders import Nominatim


# In[210]:


geolocator=Nominatim(user_agent='app')


# In[214]:


lat_lon=[]
for location in locations['Name']:
    location=geolocator.geocode(location)
    if location is None:
        lat_lon.append(np.nan)
    else:
        geo=(location.latitude,location.longitude)
        lat_lon.append(geo)


# In[215]:


locations['geo_loc']=lat_lon


# In[216]:


locations.head()


# In[217]:


Rest_locations=pd.DataFrame(df['location'].value_counts().reset_index())


# In[218]:


Rest_locations


# In[225]:


Rest_locations.columns=['Name','count']


# In[226]:


Rest_locations


# In[227]:


locations.shape


# In[228]:


Rest_locations.shape


# In[229]:


Restaurant_locations=Rest_locations.merge(locations,on='Name',how='left').dropna()


# In[230]:


Restaurant_locations.head()


# In[231]:


np.array(Restaurant_locations['geo_loc'])


# In[234]:


lat,lon=zip(*np.array(Restaurant_locations['geo_loc']))


# In[235]:


type(lat)


# In[238]:


Restaurant_locations['lat']=lat
Restaurant_locations['lon']=lon


# In[239]:


Restaurant_locations.head()


# In[240]:


Restaurant_locations.drop('geo_loc',axis=1,inplace=True)


# In[241]:


Restaurant_locations.head()


# In[242]:


get_ipython().system('pip install folium')


# In[243]:


import folium
from folium.plugins import HeatMap 


# In[249]:


def generatebasemap(default_location=[12.97,77.59],default_zoom_start=12):
    basemap=folium.Map(location=default_location,zoom_start=default_zoom_start)
    return basemap


# In[250]:


basemap=generatebasemap()


# In[251]:


basemap


# # Heatmap for restaurants in bengaluru

# In[253]:


HeatMap(Restaurant_locations[['lat','lon','count']].values.tolist(),zoom=20,radius=15).add_to(basemap)


# In[254]:


basemap


# In[ ]:




