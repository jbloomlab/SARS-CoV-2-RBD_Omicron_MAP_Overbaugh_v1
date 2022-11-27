# Set BA2 RBD DMS ACE2 binding and expression scores for thresholds
We want to make sure that the filters chosen for the ACE2 binding and RBD expression scores are reasonable such that spurious antibody-escpae mutations that merely fall into the antibody-escape gate due to their poor folding or expression are removed. 

But, we also want to make sure we aren't throwing out many mutations that are found in nature at reasonable numbers. 


```python
import os

from IPython.display import display, HTML

import math
import numpy as np
import pandas as pd
from scipy import stats

from plotnine import *

from dms_variants.constants import CBPALETTE

import yaml
```

Read config file


```python
with open('config.yaml') as f:
    config = yaml.safe_load(f)
```

Define input and output directories


```python
datadir = 'data'
resultsdir = config['bind_expr_filters_dir_Omicron_BA1']

os.makedirs(resultsdir, exist_ok=True)
```

Read in the new filters for DMS ACE2 binding and expression scores. 


```python
og_thresholds={'delta_bind':-2.35, 'delta_expr':-1.0}
new_thresholds={'delta_bind':config['escape_score_min_bind_mut_Omicron_BA1'], 'delta_expr':config['escape_score_min_expr_mut_Omicron_BA1']}

og_thresholds_df=pd.DataFrame.from_dict({'metric': ['delta_bind', 'delta_expr'], 'score': [-2.35,-1.0]})
new_filter_df=pd.DataFrame({'metric': ['delta_bind', 'delta_expr'], 'score':[config['escape_score_min_bind_mut_Omicron_BA1'],config['escape_score_min_expr_mut_Omicron_BA1']]})
display(HTML(new_filter_df.to_html(index=False)))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>metric</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>delta_bind</td>
      <td>-2.00000</td>
    </tr>
    <tr>
      <td>delta_expr</td>
      <td>-0.83361</td>
    </tr>
  </tbody>
</table>



```python
gisaid_counts_file = config['gisaid_mutation_counts']
dms_scores_file = config['mut_bind_expr']
og_dms_file = config['early2020_mut_bind_expr']
```

## Examine filters and GISAID counts


```python
dms_scores = (pd.read_csv(dms_scores_file).rename(columns={'position': 'site'}).query("target == 'Omicron_BA1'")
             [['target','wildtype', 'mutation', 'site', 'mutant', 'delta_bind', 'delta_expr']]
             )

display(HTML(dms_scores.head().to_html(index=False)))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>target</th>
      <th>wildtype</th>
      <th>mutation</th>
      <th>site</th>
      <th>mutant</th>
      <th>delta_bind</th>
      <th>delta_expr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Omicron_BA1</td>
      <td>N</td>
      <td>N331A</td>
      <td>331</td>
      <td>A</td>
      <td>-0.24400</td>
      <td>-0.83011</td>
    </tr>
    <tr>
      <td>Omicron_BA1</td>
      <td>N</td>
      <td>N331C</td>
      <td>331</td>
      <td>C</td>
      <td>-0.68319</td>
      <td>-1.33642</td>
    </tr>
    <tr>
      <td>Omicron_BA1</td>
      <td>N</td>
      <td>N331D</td>
      <td>331</td>
      <td>D</td>
      <td>-0.22267</td>
      <td>-0.54616</td>
    </tr>
    <tr>
      <td>Omicron_BA1</td>
      <td>N</td>
      <td>N331E</td>
      <td>331</td>
      <td>E</td>
      <td>-0.21514</td>
      <td>-0.49162</td>
    </tr>
    <tr>
      <td>Omicron_BA1</td>
      <td>N</td>
      <td>N331F</td>
      <td>331</td>
      <td>F</td>
      <td>-0.40294</td>
      <td>-1.18490</td>
    </tr>
  </tbody>
</table>



```python
gisaid_counts = (pd.read_csv(gisaid_counts_file)
                 .drop(columns=['isite', 'wildtype'])
                )

dms_scores=(dms_scores
            .merge(gisaid_counts,
                   on=['site', 'mutant'],
                   how='left',
                   validate='many_to_one',
                  )
            .fillna({'count':0,'n_countries':0, 'frequency': 0})
           )

dms_scores=dms_scores.melt(id_vars=['wildtype','mutation', 'site', 'mutant', 'count', 'n_countries', 'frequency'],
                           value_vars=['delta_bind', 'delta_expr'], 
                           var_name='metric', 
                           value_name='score',
                          )

display(HTML(dms_scores.head().to_html(index=False)))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>wildtype</th>
      <th>mutation</th>
      <th>site</th>
      <th>mutant</th>
      <th>count</th>
      <th>n_countries</th>
      <th>frequency</th>
      <th>metric</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>N</td>
      <td>N331A</td>
      <td>331</td>
      <td>A</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>delta_bind</td>
      <td>-0.24400</td>
    </tr>
    <tr>
      <td>N</td>
      <td>N331C</td>
      <td>331</td>
      <td>C</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>delta_bind</td>
      <td>-0.68319</td>
    </tr>
    <tr>
      <td>N</td>
      <td>N331D</td>
      <td>331</td>
      <td>D</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>5.178956e-07</td>
      <td>delta_bind</td>
      <td>-0.22267</td>
    </tr>
    <tr>
      <td>N</td>
      <td>N331E</td>
      <td>331</td>
      <td>E</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>delta_bind</td>
      <td>-0.21514</td>
    </tr>
    <tr>
      <td>N</td>
      <td>N331F</td>
      <td>331</td>
      <td>F</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>delta_bind</td>
      <td>-0.40294</td>
    </tr>
  </tbody>
</table>



```python
p = (ggplot(dms_scores
            # assign small numbers to things with 0 GISAID counts or missing scores so they still appear on plot 
            .replace({'count': {0: 0.1}, 'score': {np.nan: -5}})
            .replace({'delta_bind':'ACE2 binding', 'delta_expr':'RBD expression'})
           ) +
     aes('count', 'score') +
     geom_point(alpha=0.2, color='black') +
     facet_grid('~ metric') +
     scale_x_log10()+
     theme_classic() +
     geom_hline(data=new_filter_df.replace({'delta_bind':'ACE2 binding', 'delta_expr':'RBD expression'}),
                 mapping=aes(yintercept='score'),
                linetype='dashed',
                color=CBPALETTE[1])+
     theme(figure_size=(2.5 * 2, 2.5 * 1),
           strip_background=element_blank(),
           strip_text=element_text(size=12),
          ) +
     xlab('mutation counts in GISAID as of Aug. 1, 2021')+
     ylab('BA1 RBD DMS score\n(single mutants)')
     )

fig = p.draw()

plotfile = os.path.join(resultsdir, f"counts-v-score.pdf")
print(f"Saving plot to {plotfile}")
p.save(plotfile, verbose=False)
```

    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.


    Saving plot to results/bind_expr_filters/Omicron_BA1/counts-v-score.pdf


    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.



    
![png](bind_expr_filters_Omicron_BA1_files/bind_expr_filters_Omicron_BA1_12_3.png)
    



```python
def assign_count_categories(x):
    if x == 0:
        return "0"
    elif x < 10:
        return "1 to 9"
    elif x < 20:
        return "10 to 19"
    elif x < 50:
        return "20 to 49"
    else:
        return ">=50"
    
count_categories=["0", "1 to 9", "10 to 19", "20 to 49", ">=50"]

dms_scores=(dms_scores
            .assign(count_categories=lambda x: x['count'].apply(assign_count_categories),
                   )
           )

dms_scores=(dms_scores
            .assign(count_categories=lambda x: pd.Categorical(x['count_categories'],
                                                              categories=count_categories,
                                                              ordered=True
                                                             ))
           )

p = (ggplot(dms_scores.replace({'delta_bind':'ACE2 binding', 'delta_expr':'RBD expression'})) +
     aes('count_categories', 'score') +
     geom_hline(data=new_filter_df.replace({'delta_bind':'ACE2 binding', 'delta_expr':'RBD expression'}),
                 mapping=aes(yintercept='score'),
                linetype='dashed',
                color=CBPALETTE[1])+
     geom_boxplot(outlier_alpha=0.2) +
     facet_grid('~ metric') +
     theme_classic() +
     theme(figure_size=(2.5 * 2, 2.5 * 1),
           axis_text_x=element_text(angle=90),
           strip_background=element_blank(),
           strip_text=element_text(size=12),
          ) +
     xlab('mutation counts in GISAID as of Aug. 1 2021')+
     ylab('BA1 RBD DMS score')
     )

fig = p.draw()

plotfile = os.path.join(resultsdir, f"count-cat-v-score.pdf")
print(f"Saving plot to {plotfile}")
p.save(plotfile, verbose=False)
```

    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.


    Saving plot to results/bind_expr_filters/Omicron_BA1/count-cat-v-score.pdf


    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.



    
![png](bind_expr_filters_Omicron_BA1_files/bind_expr_filters_Omicron_BA1_13_3.png)
    



```python
x_min=-4.5
x_max=0.5

p = (ggplot(dms_scores.replace({'delta_bind':'ACE2 binding', 'delta_expr':'RBD expression'})) +
     aes(x='score', fill='count_categories') +
     geom_histogram(position='identity', bins=50) +
     facet_grid('~ metric') +
     scale_x_continuous(breaks=np.arange(x_min,x_max,0.5), limits=[x_min, x_max]) +
     geom_vline(data=new_filter_df.replace({'delta_bind':'ACE2 binding', 'delta_expr':'RBD expression'}),
                     mapping=aes(xintercept='score'),
                    linetype='dashed',
                    color=CBPALETTE[1])+
     theme_classic() +
     theme(figure_size=(2.5 * 2, 2.5 * 1),
           plot_title=element_text(size=14),
           axis_text_x=element_text(angle=90),
           strip_background=element_blank(),
           strip_text=element_text(size=12),
          ) +
     ylab('number of mutations')+
     xlab('BA1 RBD DMS score') +
     labs(fill='GISAID counts')
     )

fig = p.draw()

plotfile = os.path.join(resultsdir, f"count-score-histogram.pdf")
print(f"Saving plot to {plotfile}")
p.save(plotfile, verbose=False)
```

    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/layer.py:324: PlotnineWarning: stat_bin : Removed 26 rows containing non-finite values.
    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/layer.py:401: PlotnineWarning: geom_histogram : Removed 20 rows containing missing values.
    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.


    Saving plot to results/bind_expr_filters/Omicron_BA1/count-score-histogram.pdf


    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/layer.py:324: PlotnineWarning: stat_bin : Removed 26 rows containing non-finite values.
    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/layer.py:401: PlotnineWarning: geom_histogram : Removed 20 rows containing missing values.



    
![png](bind_expr_filters_Omicron_BA1_files/bind_expr_filters_Omicron_BA1_14_3.png)
    


Things I want to know:
1. Mutations that have **any** counts in nature but are missing scores
2. Mutations that have appreciable counts (>=50) in nature but very low scores
3. The scores corresponding to the 95th percentile of all mutations occurring >= 50x in nature
4. The scores of mutations to disulfide bonds


```python
print('Here are the naturally occurring mutations that are missing scores from BA1 DMS')
display(HTML(dms_scores
             .query('count >= 1')
             .query('score.isnull()', engine='python')
             [['wildtype','mutation', 'count', 'n_countries', 'frequency', 'score']]
             .drop_duplicates()
             .to_html(index=False)
            )
       )
```

    Here are the naturally occurring mutations that are missing scores from BA1 DMS



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>wildtype</th>
      <th>mutation</th>
      <th>count</th>
      <th>n_countries</th>
      <th>frequency</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>



```python
for metric in ['bind', 'expr']:
    m=f"delta_{metric}"
    score_filter=new_thresholds[m]
    print(f'Mutations with >=50 GISAID counts but with {metric} score < {score_filter}')
    display(HTML(dms_scores
                 .query('metric==@m & count >= 50 & score < @score_filter')
                 .drop_duplicates()
                 .sort_values(by='score')
                 .head(20)
                 .to_html(index=False)
                )
           )
```

    Mutations with >=50 GISAID counts but with bind score < -2.0



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>wildtype</th>
      <th>mutation</th>
      <th>site</th>
      <th>mutant</th>
      <th>count</th>
      <th>n_countries</th>
      <th>frequency</th>
      <th>metric</th>
      <th>score</th>
      <th>count_categories</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>A</td>
      <td>A419S</td>
      <td>419</td>
      <td>S</td>
      <td>313.0</td>
      <td>35.0</td>
      <td>0.000162</td>
      <td>delta_bind</td>
      <td>-3.34454</td>
      <td>&gt;=50</td>
    </tr>
    <tr>
      <td>Y</td>
      <td>Y501S</td>
      <td>501</td>
      <td>S</td>
      <td>94.0</td>
      <td>14.0</td>
      <td>0.000049</td>
      <td>delta_bind</td>
      <td>-2.95033</td>
      <td>&gt;=50</td>
    </tr>
    <tr>
      <td>Y</td>
      <td>Y501T</td>
      <td>501</td>
      <td>T</td>
      <td>71.0</td>
      <td>13.0</td>
      <td>0.000037</td>
      <td>delta_bind</td>
      <td>-2.30495</td>
      <td>&gt;=50</td>
    </tr>
  </tbody>
</table>


    Mutations with >=50 GISAID counts but with expr score < -0.83361



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>wildtype</th>
      <th>mutation</th>
      <th>site</th>
      <th>mutant</th>
      <th>count</th>
      <th>n_countries</th>
      <th>frequency</th>
      <th>metric</th>
      <th>score</th>
      <th>count_categories</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>P</td>
      <td>P521L</td>
      <td>521</td>
      <td>L</td>
      <td>50.0</td>
      <td>13.0</td>
      <td>0.000026</td>
      <td>delta_expr</td>
      <td>-1.21563</td>
      <td>&gt;=50</td>
    </tr>
    <tr>
      <td>P</td>
      <td>P491S</td>
      <td>491</td>
      <td>S</td>
      <td>98.0</td>
      <td>18.0</td>
      <td>0.000051</td>
      <td>delta_expr</td>
      <td>-1.20171</td>
      <td>&gt;=50</td>
    </tr>
    <tr>
      <td>P</td>
      <td>P499L</td>
      <td>499</td>
      <td>L</td>
      <td>83.0</td>
      <td>13.0</td>
      <td>0.000043</td>
      <td>delta_expr</td>
      <td>-1.17862</td>
      <td>&gt;=50</td>
    </tr>
    <tr>
      <td>A</td>
      <td>A352V</td>
      <td>352</td>
      <td>V</td>
      <td>156.0</td>
      <td>23.0</td>
      <td>0.000081</td>
      <td>delta_expr</td>
      <td>-1.17426</td>
      <td>&gt;=50</td>
    </tr>
    <tr>
      <td>A</td>
      <td>A411V</td>
      <td>411</td>
      <td>V</td>
      <td>53.0</td>
      <td>12.0</td>
      <td>0.000027</td>
      <td>delta_expr</td>
      <td>-1.16146</td>
      <td>&gt;=50</td>
    </tr>
    <tr>
      <td>G</td>
      <td>G413V</td>
      <td>413</td>
      <td>V</td>
      <td>56.0</td>
      <td>18.0</td>
      <td>0.000029</td>
      <td>delta_expr</td>
      <td>-1.16131</td>
      <td>&gt;=50</td>
    </tr>
    <tr>
      <td>P</td>
      <td>P507S</td>
      <td>507</td>
      <td>S</td>
      <td>59.0</td>
      <td>12.0</td>
      <td>0.000031</td>
      <td>delta_expr</td>
      <td>-1.07858</td>
      <td>&gt;=50</td>
    </tr>
    <tr>
      <td>K</td>
      <td>K356N</td>
      <td>356</td>
      <td>N</td>
      <td>125.0</td>
      <td>11.0</td>
      <td>0.000065</td>
      <td>delta_expr</td>
      <td>-1.07163</td>
      <td>&gt;=50</td>
    </tr>
    <tr>
      <td>P</td>
      <td>P373L</td>
      <td>373</td>
      <td>L</td>
      <td>285.0</td>
      <td>30.0</td>
      <td>0.000148</td>
      <td>delta_expr</td>
      <td>-1.06462</td>
      <td>&gt;=50</td>
    </tr>
    <tr>
      <td>D</td>
      <td>D405Y</td>
      <td>405</td>
      <td>Y</td>
      <td>77.0</td>
      <td>21.0</td>
      <td>0.000040</td>
      <td>delta_expr</td>
      <td>-1.06126</td>
      <td>&gt;=50</td>
    </tr>
    <tr>
      <td>Q</td>
      <td>Q414H</td>
      <td>414</td>
      <td>H</td>
      <td>678.0</td>
      <td>14.0</td>
      <td>0.000351</td>
      <td>delta_expr</td>
      <td>-1.01997</td>
      <td>&gt;=50</td>
    </tr>
    <tr>
      <td>G</td>
      <td>G485V</td>
      <td>485</td>
      <td>V</td>
      <td>84.0</td>
      <td>17.0</td>
      <td>0.000044</td>
      <td>delta_expr</td>
      <td>-1.01817</td>
      <td>&gt;=50</td>
    </tr>
    <tr>
      <td>P</td>
      <td>P463S</td>
      <td>463</td>
      <td>S</td>
      <td>403.0</td>
      <td>29.0</td>
      <td>0.000209</td>
      <td>delta_expr</td>
      <td>-0.97621</td>
      <td>&gt;=50</td>
    </tr>
    <tr>
      <td>G</td>
      <td>G482V</td>
      <td>482</td>
      <td>V</td>
      <td>100.0</td>
      <td>17.0</td>
      <td>0.000052</td>
      <td>delta_expr</td>
      <td>-0.97595</td>
      <td>&gt;=50</td>
    </tr>
    <tr>
      <td>P</td>
      <td>P426S</td>
      <td>426</td>
      <td>S</td>
      <td>205.0</td>
      <td>13.0</td>
      <td>0.000106</td>
      <td>delta_expr</td>
      <td>-0.97435</td>
      <td>&gt;=50</td>
    </tr>
    <tr>
      <td>K</td>
      <td>K378M</td>
      <td>378</td>
      <td>M</td>
      <td>81.0</td>
      <td>2.0</td>
      <td>0.000042</td>
      <td>delta_expr</td>
      <td>-0.97347</td>
      <td>&gt;=50</td>
    </tr>
    <tr>
      <td>K</td>
      <td>K356E</td>
      <td>356</td>
      <td>E</td>
      <td>116.0</td>
      <td>8.0</td>
      <td>0.000060</td>
      <td>delta_expr</td>
      <td>-0.94080</td>
      <td>&gt;=50</td>
    </tr>
    <tr>
      <td>V</td>
      <td>V401L</td>
      <td>401</td>
      <td>L</td>
      <td>321.0</td>
      <td>32.0</td>
      <td>0.000166</td>
      <td>delta_expr</td>
      <td>-0.93472</td>
      <td>&gt;=50</td>
    </tr>
    <tr>
      <td>L</td>
      <td>L513F</td>
      <td>513</td>
      <td>F</td>
      <td>99.0</td>
      <td>17.0</td>
      <td>0.000051</td>
      <td>delta_expr</td>
      <td>-0.90704</td>
      <td>&gt;=50</td>
    </tr>
    <tr>
      <td>A</td>
      <td>A411S</td>
      <td>411</td>
      <td>S</td>
      <td>1680.0</td>
      <td>37.0</td>
      <td>0.000870</td>
      <td>delta_expr</td>
      <td>-0.89338</td>
      <td>&gt;=50</td>
    </tr>
  </tbody>
</table>



```python
print('Here are the scores for mutations to disulfide bonds:')

p = (ggplot(dms_scores
            .replace({'delta_bind':'ACE2 binding', 'delta_expr':'RBD expression'})
            .assign(wildtype=lambda x: x['mutation'].str[0])
            .query('wildtype=="C" & mutant!="C"')
           ) +
     aes(x='score') + 
     geom_histogram(binwidth=0.25) +
     geom_vline(data=new_filter_df.replace({'delta_bind':'ACE2 binding', 'delta_expr':'RBD expression'}),
                     mapping=aes(xintercept='score'),
                    linetype='dashed',
                    color=CBPALETTE[1])+
     facet_wrap('~ metric') +
     theme_classic() +
     theme(figure_size=(2.5 * 2, 2.5 * 1),
           plot_title=element_text(size=14),
           axis_text_x=element_text(angle=90),
           strip_background=element_blank(),
           strip_text=element_text(size=12),
          ) +
     xlab('BA1 RBD DMS score')
     )

fig = p.draw()

plotfile = os.path.join(resultsdir, f"disulfide-histogram.pdf")
print(f"Saving plot to {plotfile}")
p.save(plotfile, verbose=False)
```

    Here are the scores for mutations to disulfide bonds:


    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.


    Saving plot to results/bind_expr_filters/Omicron_BA1/disulfide-histogram.pdf


    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.



    
![png](bind_expr_filters_Omicron_BA1_files/bind_expr_filters_Omicron_BA1_18_4.png)
    


### Get the bind and expr scores that correspond to the 5th percentile of mutations observed at least 50x in GISAID


```python
def get_filter(scores_df, metric, count_threshold, percentile):
    
    scores=(scores_df
            .query('metric==@metric & count >=@count_threshold')
            .dropna()
            )['score'].tolist()
            
    c=np.percentile(scores, percentile)
    
    return c

count_thresholds = [50]
percentiles=[1,2.5,5,10,25]

v=[]

for i in count_thresholds:
    for p in percentiles:
        t=(i,p)
        
        scores=(dms_scores)
        bind_filter=get_filter(scores, 'delta_bind', i, p)
        expr_filter=get_filter(scores, 'delta_expr', i, p)
        
        t=(i, p, bind_filter, expr_filter)
        
        v.append(t)
        

df = pd.DataFrame(v, columns =['count_threshold', 'percentile', 'bind_count', 'expr_count'])
display(HTML(df.to_html(index=False)))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>count_threshold</th>
      <th>percentile</th>
      <th>bind_count</th>
      <th>expr_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>50</td>
      <td>1.0</td>
      <td>-2.421118</td>
      <td>-1.182776</td>
    </tr>
    <tr>
      <td>50</td>
      <td>2.5</td>
      <td>-1.640508</td>
      <td>-1.161377</td>
    </tr>
    <tr>
      <td>50</td>
      <td>5.0</td>
      <td>-1.362491</td>
      <td>-1.057131</td>
    </tr>
    <tr>
      <td>50</td>
      <td>10.0</td>
      <td>-0.819488</td>
      <td>-0.904308</td>
    </tr>
    <tr>
      <td>50</td>
      <td>25.0</td>
      <td>-0.518110</td>
      <td>-0.560150</td>
    </tr>
  </tbody>
</table>



```python
og_dms_scores=(pd.read_csv(og_dms_file)
               # remove extraneous columns
               .drop(columns=['site_RBD','wildtype', 'mutation', 'mutation_RBD', 'bind_lib1', 'bind_lib2', 'expr_lib1', 'expr_lib2'])
               # rename some columns
               .rename(columns={'site_SARS2':'site', 'bind_avg':'delta_bind', 'expr_avg':'delta_expr'})
              )

display(HTML(og_dms_scores.head(2).to_html(index=False)))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>site</th>
      <th>mutant</th>
      <th>delta_bind</th>
      <th>delta_expr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>331</td>
      <td>A</td>
      <td>-0.03</td>
      <td>-0.11</td>
    </tr>
    <tr>
      <td>331</td>
      <td>C</td>
      <td>-0.09</td>
      <td>-1.26</td>
    </tr>
  </tbody>
</table>



```python
dms_scores=(dms_scores
            .merge((og_dms_scores
                    .melt(id_vars=['site', 'mutant',],
                          value_vars=['delta_bind', 'delta_expr'], 
                          var_name='metric', 
                          value_name='wuhan1dms_score',
                         )
                   ),
                   how='left',
                   on=['site', 'mutant', 'metric'],
                   validate='many_to_one'
                  )
           )
display(HTML(dms_scores.head().to_html(index=False)))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>wildtype</th>
      <th>mutation</th>
      <th>site</th>
      <th>mutant</th>
      <th>count</th>
      <th>n_countries</th>
      <th>frequency</th>
      <th>metric</th>
      <th>score</th>
      <th>count_categories</th>
      <th>wuhan1dms_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>N</td>
      <td>N331A</td>
      <td>331</td>
      <td>A</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>delta_bind</td>
      <td>-0.24400</td>
      <td>0</td>
      <td>-0.03</td>
    </tr>
    <tr>
      <td>N</td>
      <td>N331C</td>
      <td>331</td>
      <td>C</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>delta_bind</td>
      <td>-0.68319</td>
      <td>0</td>
      <td>-0.09</td>
    </tr>
    <tr>
      <td>N</td>
      <td>N331D</td>
      <td>331</td>
      <td>D</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>5.178956e-07</td>
      <td>delta_bind</td>
      <td>-0.22267</td>
      <td>1 to 9</td>
      <td>0.03</td>
    </tr>
    <tr>
      <td>N</td>
      <td>N331E</td>
      <td>331</td>
      <td>E</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>delta_bind</td>
      <td>-0.21514</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>N</td>
      <td>N331F</td>
      <td>331</td>
      <td>F</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>delta_bind</td>
      <td>-0.40294</td>
      <td>0</td>
      <td>-0.10</td>
    </tr>
  </tbody>
</table>



```python
print('Mutations from the original Wuhan-Hu-1 library that:')
print('pass bind: '+ str(len(og_dms_scores.query('delta_bind >= -2.35'))))
print('pass expr: '+ str(len(og_dms_scores.query('delta_expr >= -1.0'))))
print('pass both: '+ str(len(og_dms_scores.query('delta_bind >= -2.35 & delta_expr >= -1.0'))))
```

    Mutations from the original Wuhan-Hu-1 library that:
    pass bind: 3422
    pass expr: 2328
    pass both: 2269



```python
bind_threshold=new_thresholds['delta_bind']
expr_threshold=new_thresholds['delta_expr']
        
n_bind=len(dms_scores.query('metric=="delta_bind" & score >= @bind_threshold'))
n_expr=len(dms_scores.query('metric=="delta_expr" & score >= @expr_threshold'))

df=(dms_scores
     .pivot_table(index=['mutation', 'wildtype', 'mutant'],
                  values=['score'],
                  columns=['metric'],
                 )
     .reset_index()
       )

df.columns=['mutation', 'wildtype', 'mutant','delta_bind', 'delta_expr']

n_both=len(df
           .query('delta_bind >= @bind_threshold & delta_expr >= @expr_threshold')
          )
        
n_both_notC=len((df
                .assign(not_disulfide=lambda x: x['mutation'].str[0] != "C")
                .query('delta_bind >= @bind_threshold & delta_expr >= @expr_threshold & not_disulfide')
          ))

n_both_notC_notWT=len((df
                .assign(not_disulfide=lambda x: x['mutation'].str[0] != "C")
                .assign(not_WT=lambda x: x['wildtype']!=x['mutant'])
                .query('delta_bind >= @bind_threshold & delta_expr >= @expr_threshold & not_disulfide & not_WT')
          ))

total_muts_notC=len((df
                .assign(not_disulfide=lambda x: x['mutation'].str[0] != "C")
                .assign(not_WT=lambda x: x['wildtype']!=x['mutant'])
                .query('not_disulfide & not_WT')
          ))

print(f'BA1 SSM mutations that \npass bind: {n_bind} \npass expr: {n_expr} \npass both: {n_both} \npass both and not disulfide: {n_both_notC}')
print(f'Pass bind, expr, not disulfide, and not WT: {n_both_notC_notWT}')

print(f'Total number of possible mutations to non-disulfide sites: {total_muts_notC}')
```

    BA1 SSM mutations that 
    pass bind: 3198 
    pass expr: 2214 
    pass both: 2082 
    pass both and not disulfide: 2043
    Pass bind, expr, not disulfide, and not WT: 1850
    Total number of possible mutations to non-disulfide sites: 3667



```python
print(f'This percentage of all variants seen >=50x in GISAID are retained by the binding filter of {bind_threshold}')
print(round(100-stats.percentileofscore((dms_scores
                               .query('metric=="delta_bind" & count>=50')['score']), 
                              bind_threshold, 
                              kind='rank'
                             ),
            1
           )
     )

print(f'This percentage of all variants seen >=50x in GISAID are retained by the expression filter of {expr_threshold}')
print(round(100-stats.percentileofscore((dms_scores
                               .query('metric=="delta_expr" & count>=50')['score']), 
                              expr_threshold, 
                              kind='rank'
                             ),
            1
           )
     )


# dms_scores.query('metric=="delta_bind" & score >= @bind_threshold & count>=50')['score'].min()
```

    This percentage of all variants seen >=50x in GISAID are retained by the binding filter of -2.0
    98.4
    This percentage of all variants seen >=50x in GISAID are retained by the expression filter of -0.83361
    86.9



```python

```
