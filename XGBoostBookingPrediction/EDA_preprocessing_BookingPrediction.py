#!/usr/bin/env python
# coding: utf-8

# # EDA and preprocessing for a toy booking prediction data set
# We perform exploratory data analysis on two linked (by hotel ID) data sets. 
# 
# 
# The data set <b>booked_nights.csv</b> contains the following fields:
# <ul>
# <li> date : date that the session occurred </li>
# <li> hotel_id : unique identifier for the hotel </li>
# <li> search_day_distance : distance in days from the searched check_in and the session_date </li>
# <li> search_nights : number of nights the user searched </li>
# <li> device : device used by the user </li>
# <li> date_type : if the date was actively selected by the user or was a default selection </li>
# <li> user_country : country the user was when performed the search </li>
# <li> booked_nights : the sum of nights booked by the user </li>
# </ul>
# 
# The data set <b>hotel_detail.csv</b> contains the following fields:
# <ul>
# <li> hotel_id : unique identifier for the hotel </li>
# <li> country : country of hotel </li>
# <li> property_type : type of properety, e.g. Resort, Guest house etc. </li>
# <li> star_rating: star rating provided by reviewers </li>
# <li> number_of_reviews </li>
# <li> number_of_rooms </li>
# <li> median_total_rate : the median total rate charged by the hotel </li>
# <li> min_total_rate: the minimum total rate by the hotel </li>
# </ul>    
#   
# We check for missing values, NaNs, plot categorical features and plot a heat map of the correlation matrix of numerical variables.
# 
# Further we engineer new variables for time (dayofmonth) and combine star rating and number_of_reviews into new variable.
# 
# We one-hot-encode the categorical variables, and finally compute the Variance inflation Factors (VIF) of our predictors and clean out 
# correlated variables (according to VIF).
# 
# Finally the processed data is exported to pickles.

# In[1]:


dataset_folder = '/home/mattias/Documents/project/BookingPrediction/datasets/'
plot_folder = '/home/mattias/Documents/project/BookingPrediction/plots/'
dataset_pickle_folder = '/home/mattias/Documents/project/BookingPrediction/datasetPickles/'


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import sys


# In[3]:


print('Python version used for this notebook is {}'.format(sys.version))


# ### Load data into dataframes (note that data is given here, this notebook is an example). 

# In[4]:


booked_nights_path = dataset_folder + 'booked_nights.csv'
booked_nights = pd.read_csv(booked_nights_path)

hotel_detail_path = dataset_folder + 'hotel_detail.csv'
hotel_detail = pd.read_csv(hotel_detail_path)


# In[5]:


booked_nights.head()


# In[6]:


hotel_detail.head()


# ### Merge data into single data frame on id column

# In[7]:


df_ses = pd.merge(booked_nights,hotel_detail, left_on='hotel_id', right_on='hotel_id', how='left').drop('hotel_id', axis=1)


# ### Determine the frequency of NAN's in data

# In[8]:


null = df_ses.isna().sum()/len(df_ses)
null[null > 0].sort_values()


# ### Replace nan's with 0's in booked nights and number of reviews 

# #### no booked nights is coded as empty field in the csv-file, and this becomes NaN when reading file.

# In[9]:


df_ses['booked_nights'] = df_ses['booked_nights'].fillna(0)


# #### no number_of_reviews is coded as empty field in the csv-file, and this becomes NaN when reading file.

# In[10]:



df_ses['number_of_reviews'] = df_ses['number_of_reviews'].fillna(0)


# ### Drop nan entries of user_country, country, star_rating as these are a small part of the total data

# In[11]:


df_ses = df_ses.dropna(subset=['user_country','country','star_rating','number_of_rooms'])


# In[12]:


df_ses.head()


# #### List data types in dataframe

# In[13]:


df_ses.dtypes


# #### Cast user_country, country and property_type to str as this makes further processing easier: no mixed datatype errors.

# In[14]:


df_ses['user_country'] = df_ses['user_country'].astype(str)


# In[15]:


df_ses['country'] = df_ses['country'].astype(str)


# In[16]:


df_ses['property_type'] = df_ses['property_type'].astype(str)


# In[17]:


df_ses['property_type'].unique()


# ### Transform date to datetime object WEEKDAY

# In[18]:


df_ses['WEEKDAY'] = pd.to_datetime(df_ses['date']).dt.dayofweek


# ### Plot weekday histogram

# In[19]:


df_ses['WEEKDAY'].hist(bins=14)
plt.tight_layout()
plt.savefig(plot_folder + 'WEEKDAY.png',dpi=300)


# ### Transform date to datetime object dayofmonth

# In[20]:


df_ses['dayofmonth'] = pd.to_datetime(df_ses['date']).dt.day


# ### Plot dayofmonth histogram

# In[21]:


df_ses['dayofmonth'].hist(bins=60)
plt.tight_layout()
plt.savefig(plot_folder + 'dayofmonth.png',dpi=300)

# in the beginning of the month there is a drop in bookings


# ### Display frequences of booked nights values. Since the data is very skewed it is preferred to display the counts, instead of a histogram (where small values would be invisible).

# In[22]:


df_ses.booked_nights.value_counts()


# ### remove half-night values as these are suspected errors

# In[23]:


df_ses = df_ses[df_ses['booked_nights'].isin([0.5,1.5,2.5,3.5]) == False]


# In[24]:


df_ses.columns.values


# ### drop date and WEEKDAY as they have been replaced by dayofmonth

# In[25]:


df_ses = df_ses.drop(['date','WEEKDAY'],axis=1)


# ### Examine numerical variable in dataset 

# In[26]:


df_ses.describe()


# ### create vector of numerical variable names for later plots

# In[27]:


numerical = [
  'search_day_distance', 'search_nights', 'booked_nights',
       'star_rating', 'number_of_reviews', 'number_of_rooms',
       'median_total_rate', 'min_total_rate','dayofmonth'
]


# ### Plot booked nights histogram. Data is zero-inflated and so we will use a Tweedie distribution in building our predictor, see TrainBookingPredictor.ipynb
# 

# In[28]:


sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
sns.distplot(
    df_ses['booked_nights'], norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1}
).set(xlabel='booked_nights', ylabel='Count');
plt.savefig(plot_folder + 'bookednights_count.png',dpi=300)


# ### Check that no nonsense categorical values are left in data frame

# In[29]:


df_ses.star_rating.value_counts()


# In[30]:


df_ses.device.value_counts()


# In[31]:


df_ses.date_type.value_counts()


#  ### Since the data is very skewed it is preferred to display the counts, instead of a histogram (where small values would be invisible).

# In[32]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df_ses.country.value_counts())


# In[33]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df_ses.user_country.value_counts())


# In[34]:


df_ses.property_type.value_counts()


# ### Create a dataframe with non-zero booked_nights to see what influences a performed booking

# In[35]:


df_ses_nonzero_booking = df_ses[df_ses.booked_nights>0]


# ### Plot categorical variables against booked nights
# 
# #### Among our categorical variables several show a (visual) association to booked nights: country, user country, date type and property type. 
# #### The device type however does not seem to have a clear association.

# In[65]:


sns.set_context("paper", font_scale=2)  
plt.figure(figsize=(40,20))
chart = sns.boxplot(x='star_rating', y='booked_nights', data=df_ses_nonzero_booking)
plt.show()
plt.savefig(plot_folder + 'bookednights_vs_starrating.png',dpi=300)  


# In[37]:


sns.set_context("paper", font_scale=2)  
plt.figure(figsize=(40,20))
chart = sns.boxplot(x='user_country', y='booked_nights', data=df_ses_nonzero_booking)
chart.set_xticklabels(
    chart.get_xticklabels(), 
    rotation=90, 
    fontweight='light'
)
plt.savefig(plot_folder + 'bookednights_vs_usercountry.png',dpi=300)      


# In[38]:


sns.set_context("paper", font_scale=5)  
plt.figure(figsize=(40,20))
chart = sns.boxplot(x='device', y='booked_nights', data=df_ses_nonzero_booking)
chart.set_xticklabels(
    chart.get_xticklabels(), 
    fontweight='light'
)
plt.savefig(plot_folder + 'bookednights_vs_device.png',dpi=300)      


# In[39]:


sns.set_context("paper", font_scale=5)  
plt.figure(figsize=(40,20))
chart = sns.boxplot(x='date_type', y='booked_nights', data=df_ses_nonzero_booking)
chart.set_xticklabels(
    chart.get_xticklabels(), 
     fontweight='light'
)
plt.savefig(plot_folder + 'bookednights_vs_datetype.png',dpi=300)  


# In[40]:


sns.set_context("paper", font_scale=2)  
plt.figure(figsize=(40,20))
chart = sns.boxplot(x='country', y='booked_nights', data=df_ses_nonzero_booking)
chart.set_xticklabels(
    chart.get_xticklabels(), 
     fontweight='light',
    rotation=90
)
plt.savefig(plot_folder + 'bookednights_vs_country.png',dpi=300)  


# In[41]:


sns.set_context("paper", font_scale=2)  
plt.figure(figsize=(40,20))
chart = sns.boxplot(x='property_type', y='booked_nights', data=df_ses_nonzero_booking)
chart.set_xticklabels(
    chart.get_xticklabels(), 
     fontweight='light',
    rotation=90
)
plt.savefig(plot_folder + 'bookednights_vs_propertytype.png',dpi=300)  


# ### Compute correlation matrix for numerical variables (non-zero booking data frame) and display as heatmap.
# It is clear that among our variables not all are highly correlated with our re-
# sponse (booked nights). Again, note that we are only looking at actual bookings
# made.
# We thus have many independent variables which are weakly correlated to
# our response. This makes a boosting approach appropriate as it can leverage
# many weak learners together to make a strong prediction [1].
# 
# [1] Jerome H. Friedman. Greedy function approximation: A gradient boosting
# machine. Annals of Statistics, 29:1189{1232, 2000.
# 

# In[42]:


plt.figure(figsize=(40,20))
sns.heatmap(df_ses_nonzero_booking[numerical].corr(),annot=True)
plt.savefig(plot_folder + 'bookednights_vs_corrcoef.png',dpi=300)  


# ### one-hot-encoding for xgboost model of categorical variables

# In[43]:


df = pd.get_dummies(df_ses.user_country, drop_first=True,prefix='user_country')


# In[44]:


df_ses = pd.concat([df_ses,df], axis=1)


# In[45]:


df = pd.get_dummies(df_ses.country, drop_first=True,prefix='country')


# In[46]:


df_ses = pd.concat([df_ses,df], axis=1)


# In[47]:


df = pd.get_dummies(df_ses.property_type, drop_first=True,prefix='property_type')


# In[48]:


df_ses = pd.concat([df_ses,df], axis=1)


# In[49]:


df = pd.get_dummies(df_ses.device, drop_first=True,prefix='device')


# In[50]:


df_ses = pd.concat([df_ses,df], axis=1)


# In[51]:


from sklearn.preprocessing import LabelBinarizer
lb_datetype = LabelBinarizer()
lb_results = lb_datetype.fit_transform(df_ses["date_type"])
df_ses['date_type_ind'] = lb_results


# ### drop orginal category variables which have been one-hot-encoded

# In[52]:


df_ses_clean = df_ses.drop(['device','date_type','user_country','country','property_type'], axis = 1) 


# ### Create new variable num_reviews_times_star

# In[53]:


df_ses_clean['num_reviews_times_star'] = df_ses_clean['number_of_reviews']*df_ses_clean['star_rating']


# ### drop variables which were used to create num_reviews_times_star

# In[54]:


df_ses_clean = df_ses_clean.drop(['number_of_reviews','star_rating'],axis=1)


# ### create training and testing data frame. We set aside 20 percent of the data as independent testing data, and use 80 percent as training data.

# In[55]:


mask = np.random.rand(len(df_ses_clean)) < 0.8
df_train = df_ses_clean[mask]
df_test = df_ses_clean[~mask]
print('Training data set length='+str(len(df_train)))
# Set aside data for independent testing
print('Testing data set length='+str(len(df_test)))


# ### Create independent predictors and response dataframe

# In[56]:


X_test = df_test.drop(['booked_nights'],axis=1)
y_test = df_test['booked_nights']


# ### Create training data frame of predictors X and response y

# In[57]:


X_train = df_train.drop(['booked_nights'],axis=1)
y_train = df_train['booked_nights']


# ### create function to calculate variance inflation

# In[58]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)


# In[59]:


import pandas as pd
X_VIF = pd.read_pickle("./VIF.pkl")


# ### Calculate Variance Inflation Factors

# In[ ]:


X_VIF = calc_vif(X_train)


# ### List variables that have a variance inflation smaller than 5

# In[60]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):  
    X_VIF_tmp = X_VIF[X_VIF['VIF']<5]
    print(X_VIF_tmp.sort_values('VIF'))


# ### Remove all variables with VIF > 5

# In[61]:


drop_vect = X_VIF[X_VIF['VIF']>5].variables.values


# In[62]:


X_train = X_train.drop(drop_vect,axis=1)


# In[63]:


X_test = X_test.drop(drop_vect,axis=1)


# ### Save processed test and training data to pickles for use in TrainBookingPredictor.ipynb

# In[64]:


X_train.to_pickle(dataset_pickle_folder + 'postEDA_train_predictors.pkl')
X_test.to_pickle(dataset_pickle_folder + 'postEDA_independ_test_predictors.pkl')
y_train.to_pickle(dataset_pickle_folder + 'train_response.pkl')
y_test.to_pickle(dataset_pickle_folder + 'independ_test_response.pkl')


# In[ ]:




