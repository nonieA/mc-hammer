import pandas as pd
import json
import os

breast_cancer = pd.read_csv('data/raw/test_data/breast-cancer-wisconsin.DATA')
breast_cancer2 = pd.read_csv('data/raw/test_data/wdbc.DATA')

with open('data/raw/test_data/ecoli.DATA') as f:
    ecoli = f.readlines()

name_list = ['ID','mcg','gvh','lip','chg','aac','alm1','alm2','Y']

ecoli_dict = {name_list[i]:[j.split()[i] for j in ecoli] for i in range(9)}

ecoli_df = pd.DataFrame(ecoli_dict).drop(columns='ID')
ecoli_df.to_csv('data/raw/test_data/ecoli.csv')

with open('data/raw/test_data/glass.DATA') as f:
    glass = f.readlines()

glass_names = ['ID','RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Y']
def to_df(data_list,names,split,drop_n):
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
glass_df.to_csv('data/raw/test_data/glass.csv',index = False)

with open('data/raw/test_data/iris.DATA') as f:
    iris = f.readlines()
iris.pop( -1)
iris_names = ['sepel_length','sepal_width','petal_length','petal_width','Y']
iris_df = to_df(iris,iris_names,',',True)
iris_df.to_csv('data/raw/test_data/iris.csv',index = False)

with open('data/raw/test_data/magic04.DATA') as f:
    magic = f.readlines()

magic_names = ['fLength','fWidth','fSize', 'fConc', 'fConc1','fAsym','fM3Long','fM3Trans','fAlpha','fDist','Y']
magic_df = to_df(magic,magic_names,',',True)
magic_df.to_csv('data/raw/test_data/magic.csv',index=False)
with open('data/raw/test_data/wine.DATA') as f:
    wine = f.readlines()

wine_names = ['Y', 'Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids',
              'Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']

wine_df = to_df(wine,wine_names,',',True)
wine_df.to_csv('data/raw/test_data/wine.csv',index=False)

with open('data/raw/test_data/yeast.DATA') as f:
    yeast = f.readlines()
yeast = [re.sub(' +',',',i) for i in yeast]
yeast_names = ['ID','mcg','gvh','alm','mit','erl','pox','vac','nuc','Y']
yeast_df = to_df(yeast,yeast_names,',',True)
yeast_df.to_csv('data/raw/test_data/yeast.csv',index=False)