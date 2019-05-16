#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'project\machine_learning'))
	print(os.getcwd())
except:
	pass
#%%


import pandas as pd

pacients = pd.read_csv("../data/output1000/patients.csv", encoding="ISO-8859-1")
allergies = pd.read_csv("../data/output1000/allergies.csv")

#%%
pacients.head()

#%%
allergies.head()

#%%
data_raw = pd.merge(pacients, allergies, left_on='Id', right_on='PATIENT').get(['Id', 'BIRTHDATE', 'DEATHDATE', 'MARITAL', 'RACE', 'ETHNICITY', 'GENDER', 'CITY', 'CODE', 'DESCRIPTION', 'START', 'STOP'])
data_raw.head()

#%%
data_raw['BIRTHDATE_YEAR'] = data_raw['BIRTHDATE'].apply(lambda x: int(x[:4]))
data_raw['START_YEAR'] = data_raw['START'].apply(lambda x: int(x[:4]))
data_raw.head()

#%%
data_raw.corr()

#%%
data_raw['START_YEAR'].value_counts().plot(kind='bar')

#%%
data_raw['BIRTHDATE_YEAR'].value_counts().plot(kind='bar')

#%%
def month_difference(lower, upper):
  months_upper = int(upper[:4])*12 + int(upper[5:7])
  months_lower = int(lower[:4])*12 + int(lower[5:7])
  return months_upper - months_lower

data_raw['MONTHS_ALLERGY_START'] = data_raw[['START','BIRTHDATE']].apply(lambda x: month_difference(x['BIRTHDATE'], x['START']), axis = 1)
data_raw.head()
#%%
data_raw['MONTHS_ALLERGY_START'].value_counts().plot(kind='bar')

#%%
data_raw['MONTHS_ALLERGY_START'].value_counts().hist(bins = 8)
