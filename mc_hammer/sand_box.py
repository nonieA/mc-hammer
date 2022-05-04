import pandas as pd
import json
import os
import re
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles
from mc_hammer.null_distributions import pca_trans, min_max
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

breast_cancer = pd.read_csv('data/raw/breastcancerdata.csv')
breast_cancer = breast_cancer.drop(columns = ['id','Unnamed: 32'])
breast_cancer = breast_cancer.rename(columns = {'diagnosis':'Y'})
breast_cancer.to_csv('data/raw/real_data_sets/breast_cancer.csv',index = False)

with open('data/raw/ecoli.DATA') as f:
    ecoli = f.readlines()

name_list = ['ID','mcg','gvh','lip','chg','aac','alm1','alm2','Y']

ecoli_dict = {name_list[i]:[j.split()[i] for j in ecoli] for i in range(9)}

ecoli_df = pd.DataFrame(ecoli_dict).drop(columns='ID')
ecoli_df.to_csv('data/raw/real_data_sets/ecoli.csv',index = False)

with open('data/raw/glass.DATA') as f:
    glass = f.readlines()

glass_names = ['ID','RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Y']
def to_df(data_list,names,split=',',drop_n='False'):
    data_dict = {names[i]: [j.split(split)[i] for j in data_list] for i in range(len(names))}
    data_df = pd.DataFrame(data_dict)
    if 'ID' in names:
        data_df = data_df.drop(columns='ID')
    if drop_n:
        last_col = names[-1]
        data_df[last_col] = data_df[last_col].apply(lambda x:re.sub('\\n','',x))
    return(data_df)

glass_df = to_df(glass,glass_names)
glass_df['Y'] = glass_df['Y'].apply(lambda x: re.sub('\\n','',x))
glass_df.to_csv('data/raw/real_data_sets/glass.csv',index = False)

with open('data/raw/iris.DATA') as f:
    iris = f.readlines()
iris.pop( -1)
iris_names = ['sepel_length','sepal_width','petal_length','petal_width','Y']
iris_df = to_df(iris,iris_names,',',True)
iris_df.to_csv('data/raw/real_data_sets/iris.csv',index = False)

with open('data/raw/wine.DATA') as f:
    wine = f.readlines()

wine_names = ['Y', 'Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids',
              'Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']

wine_df = to_df(wine,wine_names,',',True)
wine_df.to_csv('data/raw/real_data_sets/wine.csv',index=False)

with open('data/raw/yeast.DATA') as f:
    yeast = f.readlines()
yeast = [re.sub(' +',',',i) for i in yeast]
yeast_names = ['ID','mcg','gvh','alm','mit','erl','pox','vac','nuc','Y']
yeast_df = to_df(yeast,yeast_names,',',True)
yeast_df.to_csv('data/raw/real_data_sets/yeast.csv',index=False)


circles = make_circles(n_samples=200,)

blob_1 = make_classification(
        n_samples = 100,
        n_features = 10,
        n_informative = 5,
        n_redundant = 5,
        n_classes = 3,
        n_clusters_per_class = 1,
        class_sep = 0.5,
        random_state = 2
    )[0]

pca_blob1 = min_max(blob_1,2)

blob_5 =  make_classification(
        n_samples = 100,
        n_features = 10,
        n_informative = 9,
        n_redundant = 1,
        n_classes = 3,
        n_clusters_per_class = 1,
        class_sep = 0.5,
        random_state = 2
    )[0]

pca_blob5 = min_max(blob_5,2)

plt.scatter(blob_1[:,0],blob_1[:,1])
plt.show()
plt.scatter(pca_blob1[:,0],pca_blob1[:,1])
plt.show()


plt.scatter(blob_5[:,0],blob_5[:,1])
plt.show()
plt.scatter(pca_blob5[:,0],pca_blob5[:,1])
plt.show()