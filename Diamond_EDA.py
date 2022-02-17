# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 00:18:39 2018
@filename: Diamonds-EDA-VDA
@dataset: diamonds-m.csv
@author: cyruslentin
"""

# hides all warnings
import warnings
warnings.filterwarnings('ignore')

# imports
# pandas 
import pandas as pd
# numpy
import numpy as np
# matplotlib 
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('ggplot')
# seaborn
import seaborn as sns
# utils
import utils

##############################################################
# Read Data 
##############################################################
df = pd.read_csv('diamonds-m.csv')
depVars = 'price'

##############################################################
# structure
##############################################################
print("\n*** Structure ***")
print(df.info())


##############################################################
# data type of columns
##############################################################
print("\n*** Data Type ***")
print(df.dtypes)
# print(dsRet)
# # store for future use
# dsDataType = dsRet


##############################################################
# Length Of Alpha Numeric Cols
##############################################################
dfTmp = pd.DataFrame()
dsRet = pd.Series() 
lColList = df.columns.to_list()
for vCol in lColList:
    if df[vCol].dtypes != 'object':
        continue
    dfTmp["tmp"] = df[vCol].str.len()
    lColLen = max(dfTmp['tmp'])
    dsRet[vCol] = lColLen
print("\n*** Lenght Of Alpha Cols ***")
print(dsRet)
    

"""
Precision is the number of digits in a number. 
Scale is the number of digits to the right of the decimal point in a number. 
For example, the number 123.45 has a precision of 5 and a scale of 2. 
"""
##############################################################
# Precision Of Numeric Cols
##############################################################
dfTmp = pd.DataFrame()
dsRet = pd.Series() 
lColList = df.columns.to_list()
for vCol in lColList:
    if (df[vCol].dtypes != 'int64' and df[vCol].dtypes != 'float64' ):
        continue
    #print(vCol, df[vCol].dtypes)
    dfTmp[vCol] = df[vCol].astype(str)
    dfTmp[vCol] = dfTmp[vCol].str.len()
    lColLen = max(dfTmp[vCol])
    if (df[vCol].dtypes == 'int64'):
       dsRet[vCol] = lColLen 
    else:
       dsRet[vCol] = lColLen - 1
print("\n*** Precision Of Numeric Cols ***")
print(dsRet)


##############################################################
# Scale Of Numeric Cols
##############################################################
dfTmp = pd.DataFrame()
dsRet = pd.Series() 
lColList = df.columns.to_list()
for vCol in lColList:
    if (df[vCol].dtypes != 'int64' and df[vCol].dtypes != 'float64' ):
        continue
    if df[vCol].dtypes == 'int64':
         dsRet[vCol] = 0
         continue
    #print(vCol, df[vCol].dtypes)
    dfTmp = df[[vCol]]
    dfTmp = dfTmp.fillna(-1)
    dfTmp[vCol] = dfTmp[vCol] - dfTmp[vCol].astype(int) 
    dfTmp[vCol] = df[vCol].astype(str)
    dfTmp[vCol] = dfTmp[vCol].str.len()
    lColLen = max(dfTmp[vCol])
    dsRet[vCol] = lColLen - 1
print("\n*** Scale Of Numeric Cols ***")
print(dsRet)


##############################################################
# Null Values
##############################################################
print("\n*** Null Values ***")
print(df.isnull().sum())


##############################################################
# Zero Values
##############################################################
print("\n*** Zero Values ***")
print((df==0).sum())


##############################################################
# Group By Counts Of AlphaNum Cols
# to check obvious errors
##############################################################
dfRet = pd.DataFrame()
print("\n*** Group By Counts Of Alpha Cols ***")
lColList = df.columns.to_list()
for vCol in lColList:
    if df[vCol].dtypes != 'object':
        continue
    print('\nClass Column:',vCol)
    dfRet = df.groupby(vCol).size()
    dfRet = dfRet.reset_index()
    dfRet.columns = ['Category         ','Count']
    print(dfRet)


##############################################################
# Column Data Alpha Contents ???
##############################################################
dfTmp = pd.DataFrame()
dsRet = pd.Series() 
lColList = df.columns.to_list()
for vCol in lColList:
    if (df[vCol].dtypes != 'object'):
        continue
    print(vCol)
    dfTmp = df[[vCol]]
    dfTmp = dfTmp.fillna('x')
    dfTmp['T'] = dfTmp[vCol].str.isnumeric()
    dsRet[vCol] = (dfTmp['T']==True).sum()    
print("\n*** Column Alpha DataTyoe Contains Numeric ***")
print(dsRet)

##############################################################
# handle zeros
##############################################################
print("\n*** Zero Values Count ***")
print((df==0).sum())
print("\n*** Handle Zeros ***")
lColList = df.columns.to_list()
for vCol in lColList:
    if df[vCol].dtype != "object":
        if (df[vCol]==0).sum() > 0:
            df[vCol] = np.where(df[vCol]==0,None,df[vCol])
            df[vCol] = pd.to_numeric(df[vCol], errors = 'coerce')
print("Done .. ")
print("\n*** Zero Values Count ***")
print((df==0).sum())


##############################################################
# replace null with mean for numeric cols
##############################################################
print('\nColumn Means')
print(df.mean())
print("\n*** Null Values Count ***")
print(df.isnull().sum())
print("\n*** Handle Nulls ***")
lColList = df.columns.to_list()
for vCol in lColList:
    if df[vCol].dtype != "object":
        if df[vCol].isnull().sum() > 0:
            df[vCol] = df[vCol].fillna(df[vCol].mean())
print("Done .. ")
print("\n*** Null Values Count ***")
print(df.isnull().sum())
print('\nColumn Means')
print(df.mean())


##############################################################
# quartile summary for numeric cols
##############################################################
print("\n*** Quartile Summary ***")
print(df.describe())

print("\n*** Range Values ***")
dsRet = pd.Series() 
lColList = df.columns.to_list()
for vCol in lColList:
    if (df[vCol].dtypes == 'object'):
        continue
    vRng = str(df[vCol].min()) + " - " + str(df[vCol].max())
    dsRet[vCol] =  vRng
print(dsRet)

print("\n*** Range Difference ***")
dsRet = pd.Series() 
lColList = df.columns.to_list()
for vCol in lColList:
    if (df[vCol].dtypes == 'object'):
        continue
    vRng = df[vCol].max() - df[vCol].min()
    dsRet[vCol] =  vRng
print(dsRet)

print("\n*** Variance ***")
print(df.var())

print("\n*** Standard Deviation ***")
print(df.std())

##############################################################
# handle outliers
##############################################################
print('\n*** Outlier Count ***')
print(utils.OutlierCount(df))

print('\n*** Outlier Index ***')
print(utils.OutlierIndex(df))

print('\n*** Outlier Values ***')
print(utils.OutlierValues(df))

print('\n*** Handle Outliers ***')
df = utils.HandleOutlier(df)

print('\n*** Outlier Count ***')
print(utils.OutlierCount(df))


#############################################################
# replace null with mode for alpha-numeric cols
##############################################################
print("\n*** Null Values Count ***")
print(df.isnull().sum())
print("\n*** Handle Null Values For AlphaNumerics ***")
lColList = df.columns.to_list()
for vCol in lColList:
    if df[vCol].dtypes != 'object':
        continue
    vMode = df[vCol].mode()
    if isinstance(vMode, pd.core.series.Series):        
        vMode = vMode[0]
    df[vCol] = np.where(df[vCol].isnull(), vMode, df[vCol])
print("Done ...")
print("\n*** Null Values Count ***")
print(df.isnull().sum())


##############################################################
# Frequency Distribution Table
##############################################################
dfRet = pd.DataFrame()
print("\n*** Frequency Distribution ***")
lColList = df.columns.to_list()
for vCol in lColList:
    if df[vCol].dtypes != 'object':
        continue
    print('\nClass Column:',vCol)
    dfRet = df.groupby(vCol).size()
    dfRet = dfRet.reset_index()
    dfRet.columns = ['Category         ','Count']
    print(dfRet)


##############################################################
# Convert Alpha Categoric To Numeric Categoric 
##############################################################
print("\n*** Convert Alpha Categoric To Numeric Categoric ***")
lColList = df.columns.to_list()
for vCol in lColList:
    if df[vCol].dtypes != 'object':
        continue
    print("\n"+vCol)
    print(df[vCol].unique())
    df[vCol] = pd.Categorical(df[vCol])
    df[vCol] = df[vCol].cat.codes
    print(df[vCol].unique())
print("Done ...")


##############################################################
# Rescaling 
##############################################################
print("\n*** Standard Deviation ***")
print(df.std())
# handle normalization if required
# print('\n*** Normalize Data ***')
# df = utils.NormalizeData(df, ['id','price'])
# print('Done ...')
# print("\n*** Standard Deviation ***")
# print(df.std())


##############################################################
# plot histograms
##############################################################
print("\n*** Histogram Plot ***")
lColList = df.columns.to_list()
for vCol in lColList:
    if (df[vCol].dtypes == 'object'): 
        continue
    #print("Col",vCol)
    colValues = df[vCol].values
    plt.figure(figsize=(10,5))
    sns.distplot(colValues, bins=7, kde=False, color='b')
    plt.title('Column %s' % vCol)
    plt.ylabel(vCol)
    plt.xlabel('Bins')
    plt.show()


##############################################################
# plot Boxplot
##############################################################
print("\n*** Box Plot ***")
lColList = df.columns.to_list()
for vCol in lColList:
    if (df[vCol].dtypes == 'object'): #and df[vCol].dtypes != 'float64' ):
        continue
    #print("Col",vCol)
    plt.figure(figsize=(10,5))
    sns.boxplot(y=df[vCol], color='b')
    plt.title('Column %s' % vCol)
    plt.ylabel(vCol)
    plt.xlabel('Bins')
    plt.show()

##############################################################
# Corelation
#############################################################

# corelation table
print("\n*** Correlation Table ***")
pd.options.display.float_format = '{:,.3f}'.format
dfc = df.corr()
print(dfc)

# heatmap
print("\n*** Heat Map ***")
plt.figure(figsize=(8,8))
ax = sns.heatmap(df.corr(), annot=True, cmap="PiYG")
# data.corr().style.background_gradient(cmap='coolwarm').set_precision(2)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom+0.5, top-0.5)
plt.show()

# multi colinearity 
print("\n*** Multi Colinearity ***")
lMulCorr = utils.MulCorrCols(dfc, 'price', False)
print(lMulCorr)        
print('Done ...')

##############################################################
# Scatter Plots
#############################################################
# plot Sscatterplot
print('\n*** Scatterplot ***')
lColList = df.columns.to_list()
lColList.remove('price')
for vCol in lColList:
    if (df[vCol].dtypes == 'object'): #and df[vCol].dtypes != 'float64' ):
        continue
    #print(colName)
    colValues = df[vCol].values
    plt.figure()
    sns.regplot(data=df, x='price', y=vCol, color= 'b', scatter_kws={"s": 5})
    plt.title('price v/s ' + vCol)
    plt.show()


##############################################################
# distribution plot
#############################################################
# distribution plot
colNames = ['cut', 'color', 'clarity', 'popularity']
print("\n*** Distribution Plot ***")
for colName in colNames:
    plt.figure()
    sns.countplot(df[colName],label="Count")
    plt.title(colName)
    plt.show()

"""
Requirement
Any columns which is one of below is not significant
- Identifier
- Nominal
- Descriptors
Assumption
Significant Columns are those columns which starts with or ends with 
the following terms in ColName
- Id
- Ser
- Named
- Desc
"""
##############################################################
# Significant Columns
##############################################################
lNSigTerms = ['id', 'ser', 'name', 'desc']
lsRet = []
lColList = df.columns.to_list()
for vCol in lColList:
    bIsNotSig = False
    for vTerm in lNSigTerms:
        if vCol.lower().startswith(vTerm) or vCol.lower().endswith(vTerm): 
            bIsNotSig = True
    if bIsNotSig == False:
        lsRet.append(vCol) 
# store for future use
lSigCols = lsRet       
dfSigCols = df[lSigCols]
print("\n*** Significant Columns ***")
for vCol in lSigCols:
    print(vCol)


##############################################################
# scatterplot
##############################################################
print("\n*** Scatter Plot ***")
lColList = dfSigCols.columns.to_list()
vColCount = len(lColList)
for i in range(0,vColCount):
    iCol = lColList[i]
    if (df[iCol].dtypes == 'object'):  
        continue
    for j in range(i+1,vColCount):
        jCol = lColList[j]
        if (df[jCol].dtypes == 'object'): 
            continue
        #print("\niCol",iCol)
        #print("jCol",jCol)
        plt.figure(figsize=(10,5))
        sns.regplot(data=df, x=iCol, y=jCol, color= 'b', scatter_kws={"s": 5})
        plt.title(iCol + ' v/s ' + jCol)
        plt.show()

