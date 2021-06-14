import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import numpy as np
from scipy.stats import spearmanr, skew, kurtosis

def df_edit(folder_name,file_name):
    df = pd.read_csv('data/processed/fullex/' + folder_name + '/' + file_name)
    noise = re.sub('.*noise-0','',file_name)
    noise = re.sub('_sep.*','',noise)
    noise = noise + '0%'
    sep = re.sub('.*sep-','',file_name)
    sep = re.sub('\.csv','',sep)
    if sep == '05':
        sep = '0.5'


    df['noise'] = noise
    df['seperation'] = sep
    return df

def change_cvi(x):
    if x == 'SD_score':
        return 'SD'
    elif x == 'S_Dbw':
        return x
    else:
        return re.sub('_','\\n',x)

def multi_k_df(folder_name):
    file_list = os.listdir('data/processed/fullex/' + folder_name)
    df_list = [df_edit(folder_name,i) for i in file_list]
    full_df = pd.concat(df_list)
    full_df = full_df.rename(columns={'Unnamed: 0':'Null Distributions'})
    full_df = full_df.groupby(['Null Distributions','noise','seperation']).mean().reset_index()
    full_df['Null Distributions'] = full_df['Null Distributions'].apply(lambda x: re.sub('_K_Means', '', x))
    id_vars = ['Null Distributions','noise','seperation']
    val_vars = [i for i in full_df.columns if i not in id_vars]
    full_df = pd.melt(full_df,id_vars = id_vars ,value_vars = val_vars,var_name = 'CVI')
    full_df['CVI'] = full_df['CVI'].apply(change_cvi)
    return full_df

def single_heatmap(*args,**kwargs):
    df = kwargs.pop('data')
    one_plot = pd.pivot(df,index = args[0],columns=args[1],values=args[2])
    one_plot = one_plot[['3','1','0.5']]
    sns.heatmap(one_plot,cmap = 'rocket_r',**kwargs).tick_params(left = False, bottom = False)

def heatmap(data):
    plt.figure(figsize=(8.3, 11.7))

    with sns.plotting_context(font_scale=0):
        g = sns.FacetGrid(
            data,
            row='CVI',
            col='Null Distributions',
            legend_out=False,
            margin_titles=True,
            height=1.05,
            aspect=1.5
        )
    cbar_ax = g.fig.add_axes([.92, .3, .02, .4])

    g.map_dataframe(single_heatmap, 'noise', 'seperation', 'value', vmin=0, vmax=100, cbar_ax=cbar_ax)
    g.set_xlabels('Separation')
    g.set_ylabels('Noise')

    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.tight_layout()
    g.fig.subplots_adjust(wspace=0, hspace=0, bottom=0)
    g.fig.subplots_adjust(right=20)
    g.tight_layout()
    return(g)

def corr_test(x,y):
    corr = spearmanr(x,y)
    if ((corr[0] > 0.5) or (corr[0] < -0.5)) and (corr[1] < 0.05):
        return True
    else:
        return False

if __name__ == '__main__':

    null_results = pd.read_csv('data/processed/fullex/null_results.csv')
    null_gauss_results = pd.read_csv('data/processed/fullex/null_gouss_results.csv')

    null_results_full = pd.concat([null_results,null_gauss_results])
    col_order = ['Unnamed: 0','huberts_gamma','norm_gamma', 'sillhouette_euclidean' ,'sillhouette_cosine','CH','DB',
       'dunn', 'SD_score', 'S_Dbw', 'IGP', 'BWC',  'CVNN']

    null_results_full = null_results_full[col_order]
    null_results_full = null_results_full.set_index('Unnamed: 0')
    null_results_full = null_results_full.T
    null_results_full = null_results_full.reset_index().rename(columns={'index':'CVI'})
    val_vars = null_results_full.columns[1:].tolist()
    null_results_long = pd.melt(
        null_results_full,
        id_vars='CVI',
        value_vars=val_vars,
        var_name='Null Distributions',
        value_name='% of times clusters found'
    )
    null_results_long['Null Distributions'] = null_results_long['Null Distributions'].apply(
        lambda x: re.sub('_K_Means','',x))
    null_results_long['CVI'] = null_results_long['CVI'].apply(change_cvi)
    pallate = sns.color_palette(['#D90368','#197278','#541388'])
    null_plot = sns.barplot(
        x = 'CVI',
        y='% of times clusters found',
        hue = 'Null Distributions',
        data=null_results_long,
        palette=pallate
    )
    plt.xticks(rotation=90)
    null_plot.axhline(5,color = 'r',linestyle = '--')
    plt.savefig('graphs/null_dist.png')
    plt.show()

    # expeririment 2 can identify clusters
    folder_name = 'k_means_pos_test'
    full_df = multi_k_df(folder_name)
    g = heatmap(full_df)
    plt.show()
    g.savefig('graphs/test_clusters.png')
    plt.show()

    #experiment 3 can idendify cluster number
    folder_name = 'k_means_sens_test'
    full_df = multi_k_df(folder_name)
    g = heatmap(full_df)
    plt.show()
    g.savefig('graphs/test_clusters_number.png')


    #experiment 4 can idendify cluster number
    folder_name = 'k_means_pca_test'
    full_df = multi_k_df(folder_name)
    g = heatmap(full_df)
    plt.show()
    g.savefig('graphs/test_clusters_pca.png')

    #experiment 5a
    df_list = [pd.read_csv('data/processed/fullex/k_means_skew_test/cluster_n-' + str(i) + '.csv',index_col=0) for i in [2, 4, 5]]
    skew_dicts = [{j:[skew(i[j]),kurtosis(i[j])] for j in i.columns.tolist()} for i in df_list]
    for ind,i in enumerate(skew_dicts):
        if ind == 0:
            new_skew = i
        else:
            new_skew.update(i)

    skew_df = pd.DataFrame(new_skew,index = ['skew','kurtosis']).T.reset_index()


    #experiment 5b
    df_list = [pd.read_csv('data/processed/fullex/k_means_dist_test/cluster_n-' + str(i) + '.csv') for i in [2,4,5]]
    cvi_list = df_list[0].columns.tolist()
    cvi_list = [i for i in cvi_list if i not in ['dist','Unnamed: 0']]
    corr_dict = [{j:corr_test(i[j],i['dist'])for j in cvi_list} for i in df_list]
    corr_df = pd.DataFrame(corr_dict)
    corr_df['cluster_number'] = [2,4,5]
    corr_df = corr_df[['cluster_number']+cvi_list]