#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install seaborn


# In[2]:


pip install --upgrade scikit-learn


# In[3]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# # Part 1: Data exploration

# ## Variable Identification

# In[4]:


#A breif overview of the first 5 rows in the data set
df = pd.read_csv("lending_clubFull_Data_Set.csv")
pd.set_option('display.max_columns', None)
df


# In[5]:


#Opened the Data Dictionary CSV to make it easier to understand the variables
df_Description = pd.read_csv("lendingclub_datadictionary.csv")
pd.set_option('display.max_rows', 135)
pd.set_option('max_colwidth', 200)
df_Description


# In[6]:


#There are 25000 rows and 135 columns
df.shape


# In[7]:


#A list of all the columns
list(df.columns)


# In[8]:


df.info(verbose=True, show_counts=True)


# ## Univariate Analysis 

# In[9]:


#An overview of all numerical variables 
df.describe()


# ### Loan Amount

# The data type for the variable 'loan_amnt' is a float64. This is a numerical variable which is great as it doesn't need to be transformed for the model.

# In[10]:


df['loan_amnt'].info()


# There appears to be 1 missing value in Row #23975 which shouldn't be a problem as it can be easily removed later on.

# In[11]:


print(df['loan_amnt'].isnull().sum())
df[df['loan_amnt'].isna()]


# There appears to be 23813 rows where the 'loan_amnt' appears more than once. I wouldn't say this is a big issue because its very common to take out the same sized loan as other individuals. When taking a breif look at the first 5 and last 5 duplicates we can see that other columns vary. I want to capture how a different combination of interest rates, employment length, grade, etc. while keeping the loan ammount fixed will impact the probability of an individual defaulting. Due to this reason I will keep the duplicates.

# In[12]:


print(df['loan_amnt'].duplicated().sum())
dup = df[df['loan_amnt'].duplicated(keep=False)]
dup.sort_values(by=['loan_amnt'])


# According to the Skewness and Kurtosis values, the 'loan_amnt' column exhibits a right skewed. The distribution of the loan ammount column has a short tails as all the outliers appear on the right of the upper whisker. I would like to capture how these individuals with high loan ammount behave so I will keep the outliers. 

# In[13]:


#Skewness and Kurtosis
print("Skewness: %f" % df['loan_amnt'].skew())
print("Kurtosis: %f" % df['loan_amnt'].kurt())
sns.boxplot(x=df['loan_amnt'])


# There were few individuals surveyed in some levels of loan amount. However, it does look faily balanced as there isn't any evident patterns in the count.

# In[14]:


sns.displot(df['loan_amnt'])


# In[15]:


df['loan_amnt'].describe()


# ### Interset Rate

# The data type for the variable 'int_rate' is a float64. This is a numerical variable which is great as it doesn't need to be transformed for the model.

# In[16]:


df['int_rate'].info()


# There appears to be 1 missing value in Row #23975 which shouldn't be a problem as it can be easily removed later on.

# In[17]:


df['int_rate'].isnull().sum()
df[df['int_rate'].isna()]


# There appears to be 24555 rows where the 'int_rate' appears more than once. I wouldn't say this is a big issue because its very common to take out loans with the same interest rate as other individuals. When taking a breif look at the first 5 and last 5 duplicates we can see that other columns vary. I want to capture how a different combination of of variables while keeping interest rates constant will impact the probability of an individual defaulting. Due to this reason I will keep the duplicates.

# In[18]:


print(df['int_rate'].duplicated().sum())
dup = df[df['int_rate'].duplicated(keep=False)]
dup.sort_values(by=['int_rate'])


# According to the Skewness and Kurtosis values, the 'int_rate' column exhibits a right skewed. The distribution of the interst rate column has a short tails as all the outliers appear on the right of the upper whisker. I would like to capture how these individuals with high levels on interest rates behave so I will keep the outliers.

# In[19]:


#Skewness and Kurtosis
print("Skewness: %f" % df['int_rate'].skew())
print("Kurtosis: %f" % df['int_rate'].kurt())
sns.boxplot(x=df['int_rate'])


# When looking at the interest rate, it doesn't seem like the spread is balanced as there aren't many individuals with interest rate >19%

# In[20]:


sns.displot(df['int_rate'])


# In[21]:


df['int_rate'].describe()


# ### Installment

# The data type for the variable 'installment' is a float64. This is a numerical variable which is great as it doesn't need to be transformed for the model.

# In[22]:


df['installment'].info()


# There appears to be 1 missing value in Row #23975 which shouldn't be a problem as it can be easily removed later on.

# In[23]:


df['installment'].isnull().sum()
df[df['installment'].isna()]


# There appears to be 13285 rows where the 'installment' appears more than once. I wouldn't say this is a big issue because its very common to take out loans with the same monthly payments as other individuals. When taking a breif look at the first 5 and last 5 duplicates we can see that other columns vary. I want to capture how a different combination of variables while keeping installments constant will impact the probability of an individual defaulting. Due to this reason I will keep the duplicates.

# In[24]:


print(df['installment'].duplicated().sum())
dup = df[df['installment'].duplicated(keep=False)]
dup.sort_values(by=['installment'])


# According to the Skewness and Kurtosis values, the 'installment' column exhibits a right skewed. The distribution of the interst rate column has a short tails as all the outliers appear on the right of the upper whisker. I would like to capture how these individuals with high levels of monthly payments behave so I will keep the outliers.

# In[25]:


#skewness and kurtosis
print("Skewness: %f" % df['installment'].skew())
print("Kurtosis: %f" % df['installment'].kurt())
sns.boxplot(x=df['installment'])


# In regards to monthly payments, it doesn't seem like the spread is balanced as there aren't many individuals with installments of >$600

# In[26]:


sns.displot(df['installment'])


# In[27]:


df['installment'].describe()


# ### Annual Income

# The data type for the variable 'annual_inc' is a float64. This is a numerical variable which is great as it doesn't need to be transformed for the model.

# In[28]:


df['annual_inc'].info()


# There appears to be 1 missing value in Row #23975 which shouldn't be a problem as it can be easily removed later on.

# In[29]:


print(df['annual_inc'].isnull().sum())
df[df['annual_inc'].isna()]


# There appears to be 21903 rows where the 'annual_inc' appears more than once. I wouldn't say this is a big issue because its very common to have similar incomes to other individuals especially when annual income are typically rounded. When taking a breif look at the first 5 and last 5 duplicates we can see that other columns vary. I want to capture how a different combination of variables while keeping annual income constant will impact the probability of an individual defaulting. Due to this reason I will keep the duplicates.

# In[30]:


print(df['annual_inc'].duplicated().sum())
dup = df[df['annual_inc'].duplicated(keep=False)]
dup.sort_values(by=['annual_inc'])


# According to the Skewness and Kurtosis values, the 'installment' column exhibits a right skewed. The distribution of the annual income column has a long tails as all the outliers appear on the right of the upper whisker and the Kurtosis is much greater than 3. I would like to capture how these individuals with high levels of income will behave so I will keep the outliers.

# In[31]:


#skewness and kurtosis
print("Skewness: %f" % df['annual_inc'].skew())
print("Kurtosis: %f" % df['annual_inc'].kurt())
sns.boxplot(x=df['annual_inc'])


# The annual income is spread does appear to be too balanced as there aren't many people surveyed with high incomes. The x-axis goes to 1.5 million and we can't really see the bar, but there are most likely to be 1-5 people in these bins. 

# In[32]:


plt.hist(df['annual_inc'],bins=20)


# In[33]:


df['annual_inc'].describe()


# ### Employment Length

# The data type for the variable 'emp_length' is a object. There are symbols, numbers, and letter within this column. This column will need to be altered in order to be used for the model.

# In[34]:


df['emp_length'].info()


# There appears to be 1502 missing value which shouldn't be a problem as it can be easily removed later on. 1502 rows can be removed without impacting that data too much as there are 25000 rows. 

# In[35]:


df['emp_length'].isnull().sum()


# There are 24988 duplicates within the column 'emp_length', this is understandable as this column only have values from 1 to 10

# In[36]:


print(df['emp_length'].duplicated().sum())


# Below I am converting the column into a numerical variable to perform calculations regarding the spread and distribution.

# In[37]:


df=df.replace('< 1 year','0 years')
emp_length_numeric = df.emp_length.str.extract('(^\d*)')
df['emp_length'] = emp_length_numeric
df


# In[38]:


Emp_Length_Data = df.dropna(subset=['emp_length'])
Emp_Length_Data['emp_length'] = pd.to_numeric(Emp_Length_Data['emp_length'])


# According to the Skewness and Kurtosis values, the 'emp_length' column exhibits a left skewed. Since the range of values in this column is between 1 to 10 there won't be any outliers.

# In[39]:


#skewness and kurtosis
print("Skewness: %f" % Emp_Length_Data['emp_length'].skew())
print("Kurtosis: %f" % Emp_Length_Data['emp_length'].kurt())
sns.boxplot(x=Emp_Length_Data['emp_length'])


# The data does appear to be somwhat balance when disregarding the individuals who have worked 10+ years

# In[40]:


sns.displot(df['emp_length'])
plt.xticks(rotation=45)


# In[41]:


Emp_Length_Data['emp_length'].describe()


# ### Debt to Income Ratio (DTI)

# The data type for the variable 'dti' is a float64. This is a numerical variable which is great as it doesn't need to be transformed for the model.

# In[42]:


df['dti'].info()


# There appears to be 12 missing value which shouldn't be a problem as it can be easily removed later on. 12 rows can be removed without impacting that data too much as there are 25000 rows.

# In[43]:


df['dti'].isnull().sum()


# There appears to be 21146 rows where the 'dti' appears more than once. When taking a breif look at the first 5 and last 5 duplicates we can see that other columns vary. I want to capture how a different combination of variables while keeping DTI constant will impact the probability of an individual defaulting. Due to this reason I will keep the duplicates

# In[44]:


print(df['dti'].duplicated().sum())
dup = df[df['dti'].duplicated(keep=False)]
dup.sort_values(by=['dti'])


# According to the Skewness and Kurtosis values, the 'dti' column exhibits a right skewed. The distribution of the annual income column has a long tails as all the outliers appear on the right of the upper whisker and the Kurtosis is much greater than 3. I would like to capture how these individuals with high levels of DTI will behave so I will keep the outliers.

# In[45]:


#skewness and kurtosis
print("Skewness: %f" % df['dti'].skew())
print("Kurtosis: %f" % df['dti'].kurt())
sns.boxplot(x=df['dti'])


# Most individuals appear to have a dti closer to 0 as there arevery few that have dti in the higher levels (such as >49.95)

# In[46]:


plt.hist(df['dti'],bins=20)


# In[47]:


df['dti'].describe()


# ### Grade 

# The data type for the variable 'grade' is a object. There is a letter within this column. This column will need to be altered in order to be used for the model.

# In[48]:


df['grade'].info()


# There appears to be any null values within this column which is very good.

# In[49]:


df['grade'].isnull().sum()


# There are 24992 duplicates within the column 'grade', this is understandable as this column is bounded from letter A to G or 1 to 7 once to column is transformed.

# In[50]:


print(df['grade'].duplicated().sum())


# In[51]:


df['grade'] = le.fit_transform(df['grade'])
df.head()


# There doesn't appear to be too much of a skew for the variable 'grade' as we can approximate a normal. However, there seem to be some outliers. I would like to see how individual with a lower credit grade will behave when paying back a loan. Therefore I will keep the outlier.

# In[52]:


#skewness and kurtosis
print("Skewness: %f" % df['grade'].skew())
print("Kurtosis: %f" % df['grade'].kurt())
sns.boxplot(x=df['grade'])


# The data isn't balanced as there are fewer individuals with worse grades (such as E,F,G)

# In[53]:


sns.displot(df['grade'])


# In[54]:


df['grade'].describe()


# ###  Total collection Amounts ever Owed

# The data type for the variable 'tot_coll_amt' is a float64. This is a numerical variable which is great as it doesn't need to be transformed for the model.

# In[55]:


df['tot_coll_amt'].info()


# There appears to be 1004 missing value which shouldn't be a problem as it can be easily removed later on. 1004 rows can be removed without impacting that data too much as there are 25000 rows

# In[56]:


df['tot_coll_amt'].isnull().sum()


# There appear to be 23321 duplicates. 

# In[57]:


print(df['tot_coll_amt'].duplicated().sum())


# Just as expected most of the duplicates are 0 in the column tot_coll_amt as most individuals have a $0 total collection Amounts ever Owed
# 

# In[58]:


(df['tot_coll_amt'] == 0).sum()


# According to the Skewness and Kurtosis values, the 'tot_coll_amt' column exhibits a right skewed. The distribution of the annual income column has a long tails as all the outliers appear on the right of the upper whisker and the Kurtosis is much greater than 3. I would like to capture how these individuals with high levels of total collection amounts will behave so I will keep the outliers.

# In[59]:


#skewness and kurtosis
print("Skewness: %f" % df['tot_coll_amt'].skew())
print("Kurtosis: %f" % df['tot_coll_amt'].kurt())
sns.boxplot(x=df['tot_coll_amt'])


# The 'tot_coll_amt' column is not balanced as most individuals have a total collections ammount of 0.

# In[60]:


sns.displot(df['tot_coll_amt'])


# In[61]:


df['tot_coll_amt'].describe()


# ### Number of mortgage accounts.
# 

# The data type for the variable 'mort_acc' is a float64. This is a numerical variable which is great as it doesn't need to be transformed for the model.

# In[62]:


df['mort_acc'].info()


# There appears to be 743 missing value which shouldn't be a problem as it can be easily removed later on. The 743 rows can be removed without impacting that data too much as there are 25000 rows.

# In[63]:


df['mort_acc'].isnull().sum()


# There appears to be 24980 duplicates. It would be better to see the count of each unique value within the column 'mort_acc'

# In[64]:


print(df['mort_acc'].duplicated().sum())


# Majority of the values within this column is 0 indicating the most people who were surveyed have 0 morgage accounts 

# In[65]:


ZEROS = (df['mort_acc'] == 0).sum()
ONES = (df['mort_acc'] == 1).sum()
TWOS = (df['mort_acc'] == 2).sum()
print("Number of 0's: " + str(ZEROS))
print("Number of 1's: " + str(ONES))
print("Number of 1's: " + str(TWOS))


# According to the Skewness and Kurtosis values, the 'mort_acc' column exhibits a right skewed. The distribution of the annual income column has a long tails as all the outliers appear on the right of the upper whisker and the Kurtosis is much greater than 3. I would like to capture how these individuals with more morgage accounts will behave so I will keep the outliers.

# In[66]:


#skewness and kurtosis
print("Skewness: %f" % df['mort_acc'].skew())
print("Kurtosis: %f" % df['mort_acc'].kurt())
sns.boxplot(x=df['mort_acc'])


# There are fewer individuals with 3 or more mortgages that were surveyed as most individuals who took out the loan don't have a mortgage

# In[67]:


sns.displot(df['mort_acc'])


# In[68]:


df['mort_acc'].describe()


# ### Average Current Balance 

# The data type for the variable 'avg_cur_bal' is a float64. This is a numerical variable which is great as it doesn't need to be transformed for the model.

# In[69]:


df['avg_cur_bal'].info()


# There appears to be 1004 missing value which shouldn't be a problem as it can be easily removed later on. 1004 rows can be removed without impacting that data too much as there are 25000 rows.

# In[70]:


print(df['avg_cur_bal'].isnull().sum())


# There appears to be 9266 rows where the 'avg_cur_bal' appears more than once. When taking a breif look at the first 5 and last 5 duplicates we can see that other columns vary. I want to capture how a different combination of variables while keeping bank balances constant will impact the probability of an individual defaulting. Due to this reason I will keep the duplicates.

# In[71]:


print(df['avg_cur_bal'].duplicated().sum())
dup = df[df['avg_cur_bal'].duplicated(keep=False)]
dup.sort_values(by=['avg_cur_bal'])


# According to the Skewness and Kurtosis values, the 'avg_cur_bal' column exhibits a right skewed. The distribution of the annual income column has a long tails as all the outliers appear on the right of the upper whisker and the Kurtosis is much greater than 3. I would like to capture how these individuals with high bank balances will behave so I will keep the outliers.

# In[72]:


#skewness and kurtosis
print("Skewness: %f" % df['avg_cur_bal'].skew())
print("Kurtosis: %f" % df['avg_cur_bal'].kurt())
sns.boxplot(x=df['avg_cur_bal'])


# The average currently balance appears to be decreasing within each level. This is not balanced.

# In[73]:


sns.displot(df['avg_cur_bal'])


# In[74]:


df['avg_cur_bal'].describe()


# ### Loan Status (Predicting Variable)

# The data type for the variable 'loan_status' is a object. This is our response variable which is what I hope to better predict. The column has words such as "Fully Paid", "Charged Off", "Default", etc. This column will need to be altered in order to be used for the model.

# In[75]:


df['loan_status'].info()


# There appears to be 1 missing value in Row #23975 which shouldn't be a problem as it can be easily removed later on.

# In[76]:


df['loan_status'].isnull().sum()
df[df['loan_amnt'].isna()]


# There appears to be 24990 duplicates. This makes sense as the are only 7 different values that can go in this column has they are listed below.

# In[77]:


print(df['loan_status'].duplicated().sum())


# In[78]:


print(df['loan_status'].unique())


# The data isn't balanced when looking at the values of interst which are "Fully Paid" and "Charged Off". There is a significant differnece. 

# In[79]:


sns.displot(df['loan_status'])
plt.xticks(rotation=45, ha="right")


# In[80]:


df['loan_status'].describe()


# ## Bivariate Analysis 

# In[81]:


TenFeautreDF = df[["loan_amnt", "int_rate", "installment", "annual_inc", "emp_length", 
                   "dti", "grade", "tot_coll_amt", "mort_acc", "avg_cur_bal", "loan_status"]]
TenFeautreDF


# Below I converted emp_length into a numerical variable as it consisted of symbols, text, and numbers

# In[82]:


# I noticed there was the symbol '<' in the value '< 1 year' so I wanted to replace those values with just '0 years'
TenFeautreDF=TenFeautreDF.replace('< 1 year','0 years')
#I took out the symbols and letters, which left me with just the numerical value in each row for the column 'emp_length' 
emp_length_numeric = TenFeautreDF.emp_length.str.extract('(^\d*)')
TenFeautreDF['emp_length'] = emp_length_numeric
#The column was still an object even though the rows were numeric so I converted to float using the code below
TenFeautreDF['emp_length'] = TenFeautreDF['emp_length'].astype(float)


# In[83]:


Correlation_Matrix = TenFeautreDF.corr()
Correlation_Matrix


# There appears to be some corrlation between my selected features.
# 
# Highly Corrlated Features:
# 
# **Installment and Loan Amount**
# - Monthly payments are likely to be higher when the amount of money being borrowed is bigger
# 
# **Interset Rate and Grade** 
# - Interest rates are usually determined by Credit Grade, those who have a credit grade of A are able to borrow at the prime rate which is the cheapest rate. 

# In[84]:


#correlation matrix
corrmat = TenFeautreDF.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# In[85]:


sns.set()
sns.pairplot(TenFeautreDF)
plt.show();


# ## Interseting Observations

# I initially thought that I would be seeing an increase in average current balance as the employment length increase, but this pattern isn't too visble as individuals catergorized in emp_length = 2 have a similar spread in average current balance as the individuals within emp_length = 5,6, and 9 and I found this too be quite interesting.

# In[86]:


sns.boxplot(x=df['emp_length'], y=df['avg_cur_bal'], order=['1','2','3','4','5','6','7','8','9','10'])


# I find it interesting that there is an upwards pattern in loan amount as credit scores gets worse. This is because, I felt that credit scores affect the amount of money one can borrow. For instance, an individual with a high credit (G) grade would only be able to borrow less than an individual with lower credit grade (A). However, this doesn't appear to be the case as the median loan amount increases as the grades get worse.

# In[87]:


sns.boxplot(x=df['grade'], y=df['loan_amnt'])


# The boxplot shows outlier(s) in the "Charged Off" groups as the most interesting outliers are those with an A grade, as those indiviudals are less likely to default. Also, I find it interesting that most of the individuals in the "Charged Off" group are in Grade C-D as I would expected the average to be higher. However, this could be an issue with the sampling as there were likely to be less people surveyed that belonged in the group of Grade D-G

# In[88]:


sns.boxplot(x=df['loan_status'], y=df['grade'], )
plt.xticks(rotation=45, ha="right")


# I never knew how important credit score was when determing interest rate as it appears that Grade is the sole factor in determing the interest rate you'll be paying on a loan

# In[89]:


sns.boxplot(x=df['grade'], y=df['int_rate'], )
plt.xticks(rotation=45, ha="right")


# I would assume individuals with more mortgage accounts would have a higher annual income, however the average annual income appears to be relatively constant amongst most groups. Also most of the outlier (individuals earning higher incomes) appear to have fewer mortgage accounts as well. 

# In[90]:


sns.boxplot(x=df['mort_acc'], y=df['annual_inc'])
plt.xticks(rotation=45, ha="right")


# # Part 2: Data Preparation

# In[91]:


TenFeautreDF['loan_status'].value_counts()


# I noticed that there were 2 values with 'Default' which is quite similar to 'Charged Off'

# In[92]:


#Displaying columns with "Default"
TenFeautreDF.loc[TenFeautreDF['loan_status'] == "Default"]


# I noticed that there were 5 values with 'Does not meet the credit policy. Status:Fully Paid' which is quite similar to 'Fully Paid'

# In[93]:


#Displaying columns with "Does not meet the credit policy. Status:Fully Paid"
TenFeautreDF.loc[TenFeautreDF['loan_status'] == "Does not meet the credit policy. Status:Fully Paid"].head()


# I noticed that there were 5 values with 'Does not meet the credit policy. Status:Charged Off' which is quite similar to 'Charged Off'

# In[94]:


#Displaying columns with "Does not meet the credit policy. Status:Charged Off"
TenFeautreDF.loc[TenFeautreDF['loan_status'] == "Does not meet the credit policy. Status:Charged Off"].head()


# Below I replaced the values of 'Default' and 'Does not meet the credit policy. Status:Charged Off' to 'Charged Off' and replaced the values 'Does not meet the credit policy. Status:Fully Paid",'Fully Paid' to 'Fully Paid' because I hope to capture these indiviudals as well within the model.

# In[95]:


#Replacing 'Default' with 'Charged Off'
TenFeautreDF=TenFeautreDF.replace('Default','Charged Off')
#Replacing "Does not meet the credit policy. Status:Fully Paid" with 'Fully Paid'
TenFeautreDF=TenFeautreDF.replace("Does not meet the credit policy. Status:Fully Paid",'Fully Paid')
# Replacing 'Does not meet the credit policy. Status:Charged Off' with 'Charged Off'
TenFeautreDF=TenFeautreDF.replace('Does not meet the credit policy. Status:Charged Off','Charged Off')


# In[96]:


#To check if the code above worked and replaced was used correctly
#If 3 zeros print the code has successfully replaced my desired values in the 'loan_status' column
print((TenFeautreDF['loan_status'] == 'Default').sum())
print((TenFeautreDF['loan_status'] == 'Does not meet the credit policy. Status:Fully Paid').sum())
print((TenFeautreDF['loan_status'] == 'Does not meet the credit policy. Status:Charged Off').sum())


# I only hope to predict the likeliness of a loan being 'Charged Off' or 'Fully Paid' so I will filter the data set such that the column 'loan_status' only has those 2 values

# In[97]:


# Since I hope to predict the liekliness of one fully paying back the loan or defaulting, I will filter the data set
# so that column 'loan_status' only has the values of 'Charged Off' and 'Fully Paid'
TenFeautreDF = TenFeautreDF.loc[TenFeautreDF['loan_status'].isin(['Fully Paid','Charged Off'])]
TenFeautreDF


# I have converted the 'loan_status' column into a numerical value which is required to be used for the Decision Tree Model

# In[98]:


#1 is Fully Paid, 0 is Charged Off
TenFeautreDF['loan_status'] = le.fit_transform(TenFeautreDF['loan_status'])


# ## Handling Missing Data 

# I have decided to remove any rows with a null value. It appears that there are null values in the following columns
# 
# - emp_length
# - dti
# - tot_coll_amt
# - mort_acc
# - avg_cur_bal

# In[99]:


TenFeautreDF.isnull().sum()


# In[100]:


#Removing the rows will null values
TenFeautreDF = TenFeautreDF[TenFeautreDF['emp_length'].notna()]
TenFeautreDF = TenFeautreDF[TenFeautreDF['dti'].notna()]
TenFeautreDF = TenFeautreDF[TenFeautreDF['tot_coll_amt'].notna()]
TenFeautreDF = TenFeautreDF[TenFeautreDF['mort_acc'].notna()]
TenFeautreDF = TenFeautreDF[TenFeautreDF['avg_cur_bal'].notna()]


# In[101]:


TenFeautreDF.isnull().sum()


# ## Checking Duplicates

# I have decided to keep all my duplicates because it is important to see how the likeliess of default would change when holding one variable constant while the other variable/ feature change

# In[102]:


for i in TenFeautreDF:
    x = TenFeautreDF[i].duplicated().sum()
    print (i +"| Number of Duplicates: " + str(x))


# # Part 3: Splitting data into training data and test data
# 

# In[103]:


x, y = TenFeautreDF.iloc[:, 0:10], TenFeautreDF.iloc[:,10]


# In[104]:


x


# In[105]:


y


# In[106]:


x.shape


# In[107]:


y.shape


# In[108]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y, shuffle=True)


# In[109]:


y_test.value_counts()


# In[110]:


y_train.value_counts()


# # Part 4: Evaluation

# ## Model Performance 

# In[111]:


# import ML libraries
from sklearn.tree import DecisionTreeClassifier

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion='entropy',random_state=42)

# Train Decision Tree Classifer
clf = clf.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)


# In[112]:


y_pred = clf.predict(x_test)


# Examing the confusion matrix it seems that my model appears to be having trouble categorizing the individuals as there are quite a few false negatives and positives. 
# 
# A sensitivity and specificity analysis will provide better insight 

# In[113]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

tp = float(cm[1][1])
tn = float(cm[0][0])
fp = float(cm[0][1])
fn = float(cm[1][0])

print(cm)


# The sensitivity of the current model doesnt appear to be too bad as it is 0.8001573564122738, however I do have concern regarding the specificity as it is 0.30060422960725075 indicating the the model has trouble and a tendancy of wrongfully categorizing individuals who will fully pay of the loan.

# In[114]:


# Sensitivity: the ability of a test to correctly identify loans that will default.
sensitivity = tp/(tp+fn)

# Specificity: the ability of a test to correctly identify loans that will complete(Without default).
specificity = tn/(tn+fp)

print(sensitivity)
print(specificity)


# The overall accruacy of the model is 0.6969413233458177 which is pretty awful. Hopefully we can increase the accruacy by incorporating new varaibles and removing variables that are considered userless.

# In[115]:


from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)


# The 'feature_importances_' code indicates that the features 'loan_amnt', 'dti', and 'avg_cur_bal' are mainly used by the model to distinguish individuals who are likely to defualt or fully pay off their loan. 

# In[116]:


pd.Series(clf.feature_importances_,x_train.columns)


# # Part 5: Improving the Model

# In[117]:


# I removed the last few columns from the data frame as they had many null values and won't be as much help
DF1 = df.iloc[:,3:101]
DF1


# In[118]:


#Checking the amount of null values within each column
Nulls = pd.Series(df.isnull().sum())
for key,value in Nulls.iteritems():
    print(key,",",value)


# In[119]:


#Using the code above, I removed columns with >20000 null values
DF1 = DF1.drop(columns = ['mths_since_last_delinq','mths_since_last_record','mths_since_last_major_derog' , 
                    'verification_status_joint','open_acc_6m','open_act_il','open_il_12m' ,'open_il_24m',
                    'mths_since_rcnt_il','total_bal_il','il_util','open_rv_12m','open_rv_24m','max_bal_bc' ,'all_util' , 
                    'inq_fi' ,'total_cu_tl','inq_last_12m','mths_since_recent_bc_dlq','mths_since_recent_revol_delinq',
                    'annual_inc_joint', 'desc', 'dti_joint', 'mths_since_recent_inq'], axis = 1)
DF1


# In[120]:


# Replacing 'Default' with 'Charged Off'
TenFeautreDF=TenFeautreDF.replace('Default','Charged Off')
# Replacing "Does not meet the credit policy. Status:Fully Paid" with 'Fully Paid'
TenFeautreDF=TenFeautreDF.replace("Does not meet the credit policy. Status:Fully Paid",'Fully Paid')
# Replacing 'Does not meet the credit policy. Status:Charged Off' with 'Charged Off'
TenFeautreDF=TenFeautreDF.replace('Does not meet the credit policy. Status:Charged Off','Charged Off')
# Filtering the data so that the only values in the 'loan_status' column are either "Fully Paid" or "Charged Off"
DF1 = DF1.loc[df['loan_status'].isin(['Fully Paid','Charged Off'])]
# Transforming the loan_status column to a dummy variable which represents "Fully Paid" or "Charged Off" as "1" and "0"   
DF1['loan_status'] = le.fit_transform(DF1['loan_status'])
# I noticed there was the symbol '<' in the value '< 1 year' so I wanted to replace those values with just '0 years'
DF1 = DF1.replace('< 1 year','0 years')
# Remvoing the word "years" in the column 'emp_length' so that only has numerical values
emp_length_numeric = DF1.emp_length.str.extract('(^\d*)')
DF1['emp_length'] = emp_length_numeric
# Transforming the grade column to a dummy variable which is represented by numbers
DF1['grade'] = le.fit_transform(DF1['grade'])
# Transforming the home_ownership column to a dummy variable which is represented by number
DF1['home_ownership'] = le.fit_transform(DF1['home_ownership'])
# Transforming the purpose column to a dummy variable which is represented by number
DF1['purpose'] = le.fit_transform(DF1['purpose'])
# Transforming the application_type column to a dummy variable which is represented by number
DF1['application_type'] = le.fit_transform(DF1['application_type'])
# Transforming the addr_state column to a dummy variable which is represented by number
DF1['addr_state'] = le.fit_transform(DF1['addr_state'])
# Transforming the initial_list_status column to a dummy variable which is represented by number
DF1['initial_list_status'] = le.fit_transform(DF1['initial_list_status'])
# I took out the word "years" in column 'term' so the column only consists of numerical values
DF1['term'] = DF1.term.str.extract('(\d+)')
DF1


# In[121]:


# I removed additional columns that I felt was similar to other features that I was planning to use 
DF1 = DF1.drop(columns = ['sub_grade','emp_title','verification_status' , 'pymnt_plan','title','zip_code'], axis = 1)
DF1


# In[122]:


# I set all values to 'next_pymnt_d' as I thought it was the most closest day to when the survey data was collected
# In a sense, I used this column to represent the present day
DF1['next_pymnt_d'] = '2018-01-01 00:00:00'

# Converted the objects to datetime as these columns are essentially dates
DF1['next_pymnt_d'] = pd.to_datetime(DF1['next_pymnt_d'])
DF1['last_credit_pull_d'] = pd.to_datetime(DF1['last_credit_pull_d'])
DF1['earliest_cr_line'] = pd.to_datetime(DF1['earliest_cr_line'])

# Found the amount of days between the event and the present ('2018-01-01 00:00:00') 
DF1['Days_Last_Cr_Pull'] = DF1['next_pymnt_d'] - DF1['last_credit_pull_d'] 
DF1['Days_Earliest_Cr_Line'] = DF1['next_pymnt_d'] - DF1['earliest_cr_line'] 

# Performed the same operations as above, but with the original data frame
df['next_pymnt_d'] = '2018-01-01 00:00:00'

df['next_pymnt_d'] = pd.to_datetime(df['next_pymnt_d'])
df['last_credit_pull_d'] = pd.to_datetime(df['last_credit_pull_d'])
df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])

df['Days_Last_Cr_Pull'] = df['next_pymnt_d'] - df['last_credit_pull_d'] 
df['Days_Earliest_Cr_Line'] = df['next_pymnt_d'] - df['earliest_cr_line'] 
DF1


# In[123]:


# Converted the varaible 'Days_Last_Cr_Pull' which is has a dtype datetime to a string
DF1['Days_Last_Cr_Pull'] = DF1['Days_Last_Cr_Pull'].astype(str)
# Converted the varaible 'Days_Earliest_Cr_Line' which is has a dtype datetime to a string
DF1['Days_Earliest_Cr_Line'] = DF1['Days_Earliest_Cr_Line'].astype(str)
# Took out the word "Days" in the columns so the columns only consist of numerical values
DF1['Days_Last_Cr_Pull'] = DF1.Days_Last_Cr_Pull.str.extract('(\d+)')
DF1['Days_Earliest_Cr_Line'] = DF1.Days_Earliest_Cr_Line.str.extract('(\d+)')
DF1


# In[124]:


# Removed the columns with a dtype of datetime as it can not be used in the model
DF1 = DF1.drop(columns = ['earliest_cr_line','next_pymnt_d','last_credit_pull_d'], axis = 1)
DF1


# In[125]:


# Looking at the amount of nulls in each column
Nulls = pd.Series(DF1.isnull().sum())
for key,value in Nulls.iteritems():
    print(key,",",value)


# In[126]:


# Removed the rows with a null value in any column
DF1 = DF1.dropna()
DF1


# In[127]:


# Moving the 'loan_status' column to be the 1st column in the data frame 
Columns = DF1.columns.tolist()
Loan_Status_Column =DF1.columns.get_loc("loan_status")
OrderColumns=Columns[Loan_Status_Column:Loan_Status_Column+1] + Columns[0:Loan_Status_Column] + Columns[Loan_Status_Column+1:]
DF1=DF1[OrderColumns]
DF1


# ## Evalution: Model with 66 Features 

# In[128]:


#Seting the 67 Features as x and loan_status as y
x, y = DF1.iloc[:, 1:67], DF1.iloc[:,0]


# In[129]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y, shuffle=True)


# In[130]:


# import ML libraries
from sklearn.tree import DecisionTreeClassifier

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion='entropy',random_state=42)

# Train Decision Tree Classifer
clf = clf.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)


# In[131]:


y_pred = clf.predict(x_test)


# - The amount of true positives has increased from 2034 to 2185 
# - The amount of true negatives has increased from 199 to 429 (Which is a very good news)
# - The amount of false positives has decreased from 463 to 177
# - The amount of false negatives has decreased from 508 to 187
# 
# **However, it is important to note that the last dataframe predicted 3204 individuals and the newer model only predicted for 2978. The older model predicted for 226 addidiontal individuals**

# In[132]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

tp = float(cm[1][1])
tn = float(cm[0][0])
fp = float(cm[0][1])
fn = float(cm[1][0])

print(cm)


# The sensitivity and specificity has both increased by quite a lot when incorporating the 66 Features as the sensitivity increased from  0.8001573564122738 to 0.9211635750421585. While the specificity has increased from 0.30060422960725075 to 
# 0.6969413233458177

# In[133]:


# Sensitivity: the ability of a test to correctly identify loans that will default.
sensitivity = tp/(tp+fn)

# Specificity: the ability of a test to correctly identify loans that will complete(Without default).
specificity = tn/(tn+fp)

print(sensitivity)
print(specificity)


# The overall accuracy has increased drastically as well with the incorporation of 66 Features as it was 0.6969413233458177, but now is 0.8777703156480859

# In[134]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# Obtaining the feature importances for each column will help indicate which features should be kept and help the model predict whether and individual will default or not. 
# 
# We can see below that there are some useless features as they don't help the model in any way determining whether or not an individual may default. These variables have a feature importance of 0.0. These variables can be removed.

# In[135]:


Dict = pd.Series(clf.feature_importances_,x_train.columns).to_dict()
Dict


# In[136]:


# The code below forms a list with the most important features 
Top25Features = sorted(Dict, key=Dict.get, reverse=True)[:25]
Top25Features


# In[137]:


# Making a data frame consisting of the top 25 features based on their importance based on the last model
Top25DF = df[Top25Features]
Top25DF['loan_status']= df['loan_status']
Top25DF


# In[138]:


# Performed the same operations as before
Top25DF = Top25DF.replace('Default','Charged Off')
Top25DF = Top25DF.loc[df['loan_status'].isin(['Fully Paid','Charged Off'])]
Top25DF['loan_status'] = le.fit_transform(Top25DF['loan_status'])
Top25DF['Days_Last_Cr_Pull'] = Top25DF['Days_Last_Cr_Pull'].astype(str)
Top25DF['Days_Earliest_Cr_Line'] = Top25DF['Days_Earliest_Cr_Line'].astype(str)
Top25DF['Days_Last_Cr_Pull'] = Top25DF.Days_Last_Cr_Pull.str.extract('(\d+)')
Top25DF['Days_Earliest_Cr_Line'] = Top25DF.Days_Earliest_Cr_Line.str.extract('(\d+)')
Top25DF['term'] = Top25DF.term.str.extract('(\d+)')
Top25DF['addr_state'] = le.fit_transform(Top25DF['addr_state'])
Top25DF = Top25DF.dropna()
Top25DF


# ## Evalution: Model with Top 25 Features

# In[139]:


x, y = Top25DF.iloc[:, 0:25], Top25DF.iloc[:,25]


# In[140]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y, shuffle=True)


# In[141]:


# import ML libraries
from sklearn.tree import DecisionTreeClassifier

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion='entropy',random_state=42)

# Train Decision Tree Classifer
clf = clf.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)


# In[142]:


y_pred = clf.predict(x_test)


# - The amount of true positives has increased from 2185 to 2362 
# - The amount of true negatives has increased from 429 to 517 
# - The amount of false positives has decreased from 177 to 175
# - The amount of false negatives has increased from 187 to 204 (There is a slight increase in false negatives)
# 
# **However, it is important to note that the last dataframe only predicted 2978 individuals and the newer model predicted for 3258. The newer model predicted for 280 addidiontal individuals**

# In[143]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

tp = float(cm[1][1])
tn = float(cm[0][0])
fp = float(cm[0][1])
fn = float(cm[1][0])

print(cm)


# Unfortunatly, the sensitivity and specificity haven't both increased when reducing the amount of features to 25 as the sensitivity decreased slightly 0.9211635750421585 to 0.9204988308651598. However, the specificity has increased from 0.7079207920792079 to 0.7471098265895953.

# In[144]:


# Sensitivity: the ability of a test to correctly identify loans that will default.
sensitivity = tp/(tp+fn)

# Specificity: the ability of a test to correctly identify loans that will complete(Without default).
specificity = tn/(tn+fp)

print(sensitivity)
print(specificity)


# The overall accuracy has increased slightly which is great to see as the number of features were decreased to 25 as it was 0.8777703156480859, but now is 0.883670963781461.

# In[145]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# ## Removing Correlated Features 

# In[146]:


Top25DF_2 = df[Top25Features]
Top25DF_2['Days_Last_Cr_Pull'] = Top25DF_2['Days_Last_Cr_Pull'].astype(str)
Top25DF_2['Days_Earliest_Cr_Line'] = Top25DF_2['Days_Earliest_Cr_Line'].astype(str)
Top25DF_2['Days_Last_Cr_Pull'] = Top25DF_2.Days_Last_Cr_Pull.str.extract('(\d+)')
Top25DF_2['Days_Earliest_Cr_Line'] = Top25DF_2.Days_Earliest_Cr_Line.str.extract('(\d+)')
Top25DF_2['term'] = Top25DF_2.term.str.extract('(\d+)')
Top25DF_2['addr_state'] = le.fit_transform(Top25DF_2['addr_state'])
Top25DF_2


# In[147]:


Top25DF.corr()


# In[148]:


corrmat = Top25DF.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# I plan to remove one of the features that have a correlation with another such as ... 
# - last_fico_range_high & last_fico_range_low
# - revol_util & bc_util
# - revol_bal & total_rev_hi_lim
# - tot_high_cred_lim & avg_cur_bal
# - loan_amnt & installment
# - bc_open_to_buy & total_rev_hi_lim
# 
# I will remove the feature that has less importance based on the previous findings.
# 
# 

# In[149]:


Top25DF_2 = Top25DF_2.drop(columns = ['last_fico_range_low', 'revol_util', 'total_rev_hi_lim', 'avg_cur_bal', 'loan_amnt', 'bc_open_to_buy'], axis = 1)
Top25DF_2 = Top25DF_2.dropna()
Top25DF_2['loan_status']= df['loan_status']
Top25DF_2 = Top25DF_2.loc[df['loan_status'].isin(['Fully Paid','Charged Off'])]
Top25DF_2['loan_status'] = le.fit_transform(Top25DF_2['loan_status'])
Top25DF_2


# ## Evalution: Model with Top 20 Features (After Removing Correlated Features)

# In[150]:


x, y = Top25DF_2.iloc[:, 0:19], Top25DF_2.iloc[:,19]


# In[151]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y, shuffle=True)


# In[152]:


# import ML libraries
from sklearn.tree import DecisionTreeClassifier

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion='entropy',random_state=42)

# Train Decision Tree Classifer
clf = clf.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)


# In[153]:


y_pred = clf.predict(x_test)


# - The amount of true positives has increased from 2362 to 2373 
# - The amount of true negatives has decreased from 517 to 513 
# - The amount of false positives has increased from 175 to 179
# - The amount of false negatives has decreased from 204 to 193 (There is a slight increase in false negatives)
# 
# **The preivous model and the new model both predicted the likeliness of deafult for 3258 individuals**

# In[154]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

tp = float(cm[1][1])
tn = float(cm[0][0])
fp = float(cm[0][1])
fn = float(cm[1][0])

print(cm)


# There wasn't any significant changes within the sensitivity and specificity. The previous model had a sensitivity of 0.9204988308651598 and the newer model without correlated variables had a slightly higher sensitivity of 0.9247856586126266. Also, the specificity of the prior model was 0.7471098265895953, the newer model had a slight decrease as the specificity was 0.7413294797687862   

# In[155]:


# Sensitivity: the ability of a test to correctly identify loans that will default.
sensitivity = tp/(tp+fn)

# Specificity: the ability of a test to correctly identify loans that will complete(Without default).
specificity = tn/(tn+fp)

print(sensitivity)
print(specificity)


# The overall accuracy has increased slightly which is great to see as highly correlated features were removed as it was 0.883670963781461 in the previous model, but now is 0.8858195211786372.

# In[156]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[157]:


Dict = pd.Series(clf.feature_importances_,x_train.columns).to_dict()
Dict


# In[158]:


Top25DF_2.corr()


# In[159]:


corrmat = Top25DF_2.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# ## Conclusion

# There doesn't appear to be any more correlated variables. I am content with the model which consisted of the Top 20 Features after removing all the correlated features. The accruacy is pretty good, as it is 0.8858195211786372. The feature 'last_fico_range_high' appears to be an important variable for the decision tree, indiciating that it is a crucial variable that must be included. The new column 'Days_Last_Cr_Pull' that I created appears to be an important variable as well for the decision tree. 'Days_Last_Cr_Pull' basically represents the amount of days between the last credit pull and the present. However, I feel that one thing that can be improved on is the specificity as it wasn't as high as the models sensitivity. Overall, the model does perform adequatly in predicting wheather an individual is likely to full pay off or default on a loan.
