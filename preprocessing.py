import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn import decomposition
from sklearn.preprocessing import MinMaxScaler

file = 'train.csv'


def get_raw():
	df = pd.read_csv(file)
	return(df)



def get_cut():
	df = get_raw()
	df['SalePrice'] = np.log(df['SalePrice'])

	# Get rid of features missing >10% of rows
	missing = df.isnull().sum() / df.shape[0]
	missing_list = list(missing[missing > 0.1].index)
	df_cut = df.drop(missing_list, axis=1)

	# Get other missing features
	missing = df_cut.isnull().sum() / df_cut.shape[0]
	missing = missing[missing > 0]

	# Remove significant features form new missing list
	missing_list = list(missing.index)
	missing_list.remove('BsmtQual')
	missing_list.remove('GarageYrBlt')
	df_cut = df_cut.drop(missing_list, axis=1)

	# Encode into numerical values
	df_cut['BsmtQual'] = pd.Categorical(df_cut['BsmtQual']).codes

	# Imputing missing observations
		# Impute 'GarageYrBlt' using mean
	imputer = preprocessing.Imputer(strategy='median')
	yrBlt = imputer.fit_transform(df_cut['GarageYrBlt'].reshape(-1,1))
	yrBlt = pd.DataFrame(yrBlt, columns=['GarageYrBlt'])
		# Impute BsmtQual using most frequent value
	bsmtQual = df_cut['BsmtQual'].replace({-1:3})

	# Concat dataframe together
	df_cut['BsmtQual'] = bsmtQual 
	df_cut['GarageYrBlt'] = yrBlt

	# Removing outliers based on 'SalePrice'
	outliers = np.sort(df_cut['SalePrice'])[-2:]
	df_cut = df_cut[df_cut['SalePrice'] < np.min(outliers)]


	return(df_cut)


# Returns normalized df of significant numerical features
def get_num():
	df = get_cut()

	df_num = df[['1stFlrSF', 'TotalBsmtSF', 'GarageArea',
             'YearBuilt', 'GrLivArea', 'OpenPorchSF', 'YearRemodAdd', 'GarageYrBlt']]

    # PCA
	pca = decomposition.PCA(1)
	X = pca.fit_transform(df_num[['TotalBsmtSF', '1stFlrSF']])
	X = pd.DataFrame(X)
	X.columns = ['Bsmt1stFlrPCA']
	df_num = df_num.drop(['1stFlrSF', 'TotalBsmtSF'], axis=1).reset_index()
	df_num = pd.concat([df_num, X], axis=1)

	col = df_num.columns

	scaler = preprocessing.MinMaxScaler()
	df_num = scaler.fit_transform(df_num)
	df_num = pd.DataFrame(df_num, columns = col)
	return(df_num)



def get_cat():
	df = get_cut()

	# Get significant categorical features
	# Significance determined by correlation and MIC
	df_cat = df[['Neighborhood','OverallQual','ExterQual','BsmtQual','HeatingQC','FullBath','KitchenQual','TotRmsAbvGrd', 'Fireplaces','GarageCars']]
	df_cat = pd.get_dummies(df_cat)

	return(df_cat)



def split_train_test(df, frac=0.25):
	split_index = int(np.round(df.shape[0] * frac))

	train = df.iloc[split_index:, :]
	test = df.iloc[:split_index, :]

	train_l = train['SalePrice']
	train_i = train.drop('SalePrice', axis=1)
	test_l = test['SalePrice']
	test_i = test.drop('SalePrice', axis=1)
	

	return(train_i, train_l, test_i, test_l)



def get_data():
	num = get_num().reset_index(drop=True)
	cat = get_cat().reset_index(drop=True)
	sale = get_cut()['SalePrice'].reset_index(drop=True)

	df = pd.concat([num, cat, sale], axis=1)
	df = df.drop(df.columns[0], axis=1)
	return(df)


def score(x, y):
    score = np.power(np.mean(np.square(x-y)), 0.5)
    return(score)













