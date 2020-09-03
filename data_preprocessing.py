#after generating the data files concat the files to make the dataset of 1000 movies and merge them using Title as the key. Run the commented commands in a python terminal
#df1 = pd.read_csv("D:\web\data.csv")
#df2 = pd.read_csv("D:\web\data2.csv")
#df3 = pd.read_csv("D:\web\data3.csv")
#df4 = pd.read_csv("D:\web\data4.csv")
#frames = [df1,df2,df3,df4]
#res = pd.concat(frames)
#res.to_csv("D:\web\m_es_data.csv")
#df1_p = pd.read_csv("D:\web\plotdata.csv")
#df2_p = pd.read_csv("D:\web\plotdata2.csv")
#df3_p = pd.read_csv("D:\web\plotdata3.csv")
#df4_p = pd.read_csv("D:\web\plotdata4.csv")
#frames = [df1_p,df2_p,df3_p,df4_p]
#res = pd.concat(frames)
#res.to_csv("D:\web\p_es_data.csv")
#fd = pd.read_csv("D:\web\p_es_data.csv")
#fd2 = pd.read_csv("D:\web\m_es_data.csv")
#fd3 = pd.merge(fd,fd2,on = "Title")
#fd3.to_csv("D:\web\es_data.csv")

import pandas as pd
import numpy as np
df = pd.read_csv("D:\web\es_data.csv")
df[['Genre1','Genre2','Genre3']] = df.Genre.str.split(",",expand=True)
print(df)
df ["Genre1"] = df.Genre1.replace(np.nan,"None")
df ["Genre2"] = df.Genre2.replace(np.nan,"None")
df ["Genre3"] = df.Genre3.replace(np.nan,"None")
for i in range(0,1000):
	df["Genre1"][i] = df["Genre1"][i].strip()
	df["Genre2"][i] = df["Genre2"][i].strip()
	df["Genre3"][i] = df["Genre3"][i].strip()

gen1 = df.Genre1.unique()
gen2 = df.Genre2.unique() 
gen3 = df.Genre3.unique()
print(gen1)
print(gen2)
print(gen3)
#compare all the unique values in each genre2 and genre1
def subset(arr1,arr2,n,m):
	i = 0
	j = 0
	for i in range(n):
		for j in range(m):
			if(arr1[i] == arr2[j]):
				break
		if(j == m-1):
			print(arr1[i], end = " ",)
n = len(gen2)
m = len(gen1)
subset(gen2,gen1,n,m)

#compare genre3 and genre1
n = len(gen2)
m = len(gen3)
subset(gen2,gen3,n,m)
# add Fantasy genre to gen2 to make it a supserset of all the 3 Genre
gen2 = np.append(gen2,'Fantasy')
for gen in gen2:
	df[gen] = np.where((df.Genre1  == gen) | (df.Genre2 == gen) | (df.Genre3 == gen) ,1, 0)

df = df.drop(['None'],axis = 1)
df = df.rename(columns = {'Unnamed: 0':'ID'})
df.to_csv("D:\web\es_data.csv")