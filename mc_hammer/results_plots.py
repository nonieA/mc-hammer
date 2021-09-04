import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import numpy as np
from scipy.stats import spearmanr, skew, kurtosis
import statsmodels.api as sm

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

def change_cvi(x,space=False):
    if x == 'SD_score':
        return 'SD'
    elif x == 'S_Dbw':
        return x
    else:
        if space:
            return re.sub('_', ' ', x)
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

def get_dist(x):
    if 'min_max' in x:
        return 'min_max'
    elif 'random_shuffle' in x:
        return 'random_shuffle'
    else:
        return 'pca_trans'

def remove_dunn_min(df):
    col_list = [i for i in df.columns if 'dunn_min' in i]
    df = df.drop(columns = col_list)
    df = pd.melt(df,value_vars =df.columns.to_list())
    return df

def linearmodel_test(df,var):
    x = df[[var,'k']]
    y = df['distance']
    X2 = sm.add_constant(x)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    pval =True if est2.pvalues[var] < 0.05 else False
    coef = est2.params[var]
    return {
        'metric':var,
        'pval':pval,
        'coef':coef
    }

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
    df_list = [pd.read_csv('data/processed/fullex/k_means_dist_test/cluster_n-' + str(i) + '.csv',index_col=0) for i in [2, 4, 5]]
    skew_dicts = [{j:[skew(i[j]),kurtosis(i[j])] for j in i.columns.tolist()} for i in df_list]
    for ind,i in enumerate(skew_dicts):
        if ind == 0:
            new_skew = i
        else:
            new_skew.update(i)

    skew_df = pd.DataFrame(new_skew,index = ['skew','kurtosis']).T.reset_index()
    skew_df['k'] = skew_df['index'].apply(lambda x: int(re.sub('_.*','',x)))
    skew_df['method'] = skew_df['index'].apply(lambda x:re.sub('\d_[a-z]*_[a-z]*_','',x))
    skew_df['distribution'] = skew_df['index'].apply(get_dist)
    dunn_min = skew_df[skew_df['method'] == 'dunn_min']
    skew_df = skew_df[skew_df['method'] != 'dunn_min']
    skew_df = skew_df.drop(columns = 'index')
    kurt_df = skew_df.drop(columns='skew')
    skew_df = skew_df.drop(columns = 'kurtosis')
    kurt_df = kurt_df.pivot(index = ['k','distribution'],columns = 'method',values = 'kurtosis').reset_index()
    skew_df = skew_df.pivot(index=['k', 'distribution'], columns='method', values='skew').reset_index()
    kurt_df.to_csv('data/processed/tables/kurt.csv')
    skew_df.to_csv('data/processed/tables/skew.csv')
    new_df_list = [remove_dunn_min(i) for i in df_list]
    full_df = pd.concat(new_df_list)
    full_df['k'] = full_df['variable'].apply(lambda x: int(re.sub('_.*','',x)))
    full_df['method'] = full_df['variable'].apply(lambda x:re.sub('\d_[a-z]*_[a-z]*_','',x))
    full_df['distribution'] = full_df['variable'].apply(get_dist)
    full_df = full_df.drop(columns='variable')
    full_df['distribution'] = full_df['distribution'].apply(change_cvi,space=True)
    full_df['method'] = full_df['method'].apply(change_cvi, space=True)
    g = sns.FacetGrid(
        full_df,
        col='distribution',
        row='method',
        hue='k',
        sharey=False, sharex=False,margin_titles=True,palette=pallate)
    g.map(sns.kdeplot, 'value')
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.set(yticks=[],xticks=[])
    g.set_xlabels('')
    g.add_legend()
    g.tight_layout()
    g.fig.subplots_adjust(wspace=0.1, hspace=0.1, bottom=0.1)
    plt.show()
    g.savefig('graphs/distributions.png')

    #experiment 5b
    file_list = os.listdir('data/processed/fullex/size_test')
    df_list = [pd.read_csv('data/processed/fullex/size_test/' +i ).drop(columns = ['Unnamed: 0','mult']) for i in file_list]
    size_df = pd.concat(df_list)
    gauss_df = size_df[size_df['dist']=='gauss']
    uniform_df = size_df[size_df['dist']=='uniform']
    col_list = [i for i in gauss_df.columns if i not in ['k', 'dist', 'distance']]
    gauss_res = pd.DataFrame([linearmodel_test(gauss_df,i) for i in col_list])
    uniform_res = pd.DataFrame([linearmodel_test(uniform_df,i) for i in col_list])
    dunn_min_size = pd.concat([gauss_res[gauss_res['metric'] == 'dunn_min'],uniform_res[uniform_res['metric'] == 'dunn_min']])
    gauss_res = gauss_res[gauss_res['metric'] != 'dunn_min']
    uniform_res = uniform_res[uniform_res['metric'] != 'dunn_min']
    gauss_res.to_csv('data/processed/tables/size_gauss.csv')
    uniform_res.to_csv('data/processed/tables/size_uniform.csv')
    gauss_df = gauss_df.drop(columns=['dunn_min','dist'])
    id_cols = ['k','distance']
    val_cols = [i for i in gauss_df.columns if i not in id_cols]
    gauss_df =pd.melt(gauss_df,id_vars=id_cols,value_vars=val_cols)
    gauss_df['variable'] = gauss_df['variable'].apply(change_cvi,space=True)
    uniform_df = uniform_df.drop(columns=['dunn_min','dist'])
    uniform_df = pd.melt(uniform_df,id_vars=id_cols,value_vars=val_cols)
    uniform_df['variable'] = uniform_df['variable'].apply(change_cvi,space=True)
    val_cols = [change_cvi(i,True) for i in val_cols]
    g_g = sns.lmplot(
        data=gauss_df,
        x='distance',
        y='value',
        col='variable',
        hue='k',
        col_wrap=3,
        sharey=False,
        palette = pallate,
        scatter_kws={'alpha':0.2,'linewidth':0}
    )
    axes = g_g.axes
    for idx,i in enumerate(val_cols):
        min_val = min(gauss_df['value'][gauss_df['variable']==i])*0.9
        max_val = max(gauss_df['value'][gauss_df['variable'] == i]) * 1.1
        axes[idx].set_ylim(min_val,max_val)
    g_g.set(xticks=[])
    g_g.set_titles(col_template="{col_name}")
    plt.show()
    g_g.savefig('graphs/size_gauss.png')

    g_u = sns.lmplot(
        data=uniform_df,
        x='distance',
        y='value',
        col='variable',
        hue='k',
        col_wrap=3,
        sharey=False,
        palette = pallate,
        scatter_kws={'alpha':0.2,'linewidth':0}
    )
    axes = g_u.axes
    for idx,i in enumerate(val_cols):
        min_val = min(uniform_df['value'][uniform_df['variable']==i])*0.9
        max_val = max(uniform_df['value'][uniform_df['variable'] == i]) * 1.1
        axes[idx].set_ylim(min_val,max_val)
    g_u.set(xticks=[])
    g_u.set_titles(col_template="{col_name}")
    plt.show()
    g_u.savefig('graphs/size_uniform.png')

    # experiemnt 7 dunn
    null = pd.read_csv('data/processed/fullex/dunn_results/no_clusters.csv')
    null_res = [1 if i <0.05 else 0 for i in null['res']]
    null_res = sum(null_res)/len(null_res)
    clusters = pd.read_csv('data/processed/fullex/dunn_results/clusters.csv').drop(columns='Unnamed: 0')
    clusters = clusters[clusters['k'] == 2].drop(columns = 'k')
    clusters['Cluster Condition'] = clusters.apply(lambda x: 'noise: '+str(x[1]) +'\nseperation: '+str(x[2]),axis =1)
    clusters = clusters.drop(columns = ['noise','sep'])
    clusters.loc[len(clusters)] = [null_res,'no clusters']
    clusters['results'] = clusters['results'] * 100
    pal2 = {i:'#197278' if i =='no clusters' else '#541388' for i in clusters['Cluster Condition']}
    sns.set_theme()
    g = sns.barplot(data=clusters,x='Cluster Condition',y='results',palette=pal2)
    g.set(ylabel = '% of times clusters found')
    g.axhline(5,color = 'r',linestyle = '--')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('graphs/dunn_res.png')
    plt.show()

    dunn_min_size['metric'] = dunn_min_size['metric'].apply(change_cvi,space=True)
    dunn_min_size.to_csv('data/processed/tables/dunn_size.csv')
    dunn_min = dunn_min.drop(columns=['index','method'])
    dunn_min['distribution'] = dunn_min['distribution'].apply(change_cvi,space=True)
    dunn_min.to_csv('data/processed/tables/dunn_skew.csv')
    df_list = [pd.read_csv('data/processed/fullex/k_means_dist_test/cluster_n-' + str(i) + '.csv', index_col=0) for i in
               [2, 4, 5]]
    dunn_skew = pd.concat(df_list)
    drop_cols = [i for i in dunn_skew.columns if 'dunn_min' not in i]
    dunn_skew = dunn_skew.drop(columns =drop_cols)
    dunn_skew = pd.melt(dunn_skew,value_vars =dunn_skew.columns.to_list())
    dunn_skew['k'] = dunn_skew['variable'].apply(lambda x: int(re.sub('_.*', '', x)))
    dunn_skew['distribution'] = dunn_skew['variable'].apply(get_dist)
    dunn_skew = dunn_skew.drop(columns='variable')
    dunn_skew['distribution'] = dunn_skew['distribution'].apply(change_cvi,space=True)
    sns.set_style('white')
    g = sns.FacetGrid(
        dunn_skew,
        col='distribution',
        hue='k',
        sharey=False, sharex=False,margin_titles=True,palette=pallate)
    g.map(sns.kdeplot, 'value')
    g.set_titles(col_template="{col_name}")
    g.set(yticks=[],xticks=[])
    g.set_xlabels('')
    g.add_legend()
    g.tight_layout()
    g.fig.subplots_adjust(wspace=0.1, hspace=0.1, bottom=0.1)
    plt.show()
    g.savefig('graphs/dunn_distributions.png')

    g_g = sns.lmplot(
        data=size_df,
        x='distance',
        y='dunn_min',
        col='dist',
        hue='k',
        sharey=False,
        palette=pallate,
        scatter_kws={'alpha': 0.2, 'linewidth': 0}
    )

    g_g.set(xticks=[])
    g_g.set_titles(col_template="{col_name}")
    plt.show()
    g_g.savefig('graphs/dunn_size.png')

    # clustEHR results

    clus_res = pd.read_csv('data/processed/fullex/clustEHR_res/results.csv').drop(columns = 'Unnamed: 0')
    val_vars = ['sillhouette_euclidean', 'CH', 'DB','BWC']
    id_vars = [i for i in clus_res.columns if i not in val_vars]
    clust = pd.melt(clus_res,id_vars,val_vars)
    g_g = sns.lmplot(
        data=clust,
        x='value',
        y='anova_count',
        hue='variable',
        col='n',
        sharey=False,
        scatter_kws={'alpha': 0.2, 'linewidth': 0}
    )
    plt.show()