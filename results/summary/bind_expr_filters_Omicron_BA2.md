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
resultsdir = config['bind_expr_filters_dir_Omicron_BA2']

os.makedirs(resultsdir, exist_ok=True)
```

Read in the new filters for DMS ACE2 binding and expression scores. 


```python
og_thresholds={'delta_bind':-2.35, 'delta_expr':-1.0}
new_thresholds={'delta_bind':config['escape_score_min_bind_mut_Omicron_BA2'], 'delta_expr':config['escape_score_min_expr_mut_Omicron_BA2']}

og_thresholds_df=pd.DataFrame.from_dict({'metric': ['delta_bind', 'delta_expr'], 'score': [-2.35,-1.0]})
new_filter_df=pd.DataFrame({'metric': ['delta_bind', 'delta_expr'], 'score':[config['escape_score_min_bind_mut_Omicron_BA2'],config['escape_score_min_expr_mut_Omicron_BA2']]})
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
      <td>-0.95489</td>
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
dms_scores = (pd.read_csv(dms_scores_file).rename(columns={'position': 'site'}).query("target == 'Omicron_BA2'")
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
      <td>Omicron_BA2</td>
      <td>N</td>
      <td>N331A</td>
      <td>331</td>
      <td>A</td>
      <td>-0.08339</td>
      <td>-0.62526</td>
    </tr>
    <tr>
      <td>Omicron_BA2</td>
      <td>N</td>
      <td>N331C</td>
      <td>331</td>
      <td>C</td>
      <td>-0.61624</td>
      <td>-1.18984</td>
    </tr>
    <tr>
      <td>Omicron_BA2</td>
      <td>N</td>
      <td>N331D</td>
      <td>331</td>
      <td>D</td>
      <td>-0.14670</td>
      <td>-0.53294</td>
    </tr>
    <tr>
      <td>Omicron_BA2</td>
      <td>N</td>
      <td>N331E</td>
      <td>331</td>
      <td>E</td>
      <td>-0.14146</td>
      <td>-0.37718</td>
    </tr>
    <tr>
      <td>Omicron_BA2</td>
      <td>N</td>
      <td>N331F</td>
      <td>331</td>
      <td>F</td>
      <td>-0.53604</td>
      <td>-1.12351</td>
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
      <td>-0.08339</td>
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
      <td>-0.61624</td>
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
      <td>-0.14670</td>
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
      <td>-0.14146</td>
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
      <td>-0.53604</td>
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
     ylab('BA2 RBD DMS score\n(single mutants)')
     )

fig = p.draw()

plotfile = os.path.join(resultsdir, f"counts-v-score.pdf")
print(f"Saving plot to {plotfile}")
p.save(plotfile, verbose=False)
```

    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.


    Saving plot to results/bind_expr_filters/Omicron_BA2/counts-v-score.pdf


    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.



    
![png](bind_expr_filters_Omicron_BA2_files/bind_expr_filters_Omicron_BA2_12_3.png)
    



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
     ylab('BA2 RBD DMS score')
     )

fig = p.draw()

plotfile = os.path.join(resultsdir, f"count-cat-v-score.pdf")
print(f"Saving plot to {plotfile}")
p.save(plotfile, verbose=False)
```

    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/layer.py:324: PlotnineWarning: stat_boxplot : Removed 36 rows containing non-finite values.


    Saving plot to results/bind_expr_filters/Omicron_BA2/count-cat-v-score.pdf


    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/layer.py:324: PlotnineWarning: stat_boxplot : Removed 36 rows containing non-finite values.



    
![png](bind_expr_filters_Omicron_BA2_files/bind_expr_filters_Omicron_BA2_13_3.png)
    



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
     xlab('BA2 RBD DMS score') +
     labs(fill='GISAID counts')
     )

fig = p.draw()

plotfile = os.path.join(resultsdir, f"count-score-histogram.pdf")
print(f"Saving plot to {plotfile}")
p.save(plotfile, verbose=False)
```

    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/layer.py:324: PlotnineWarning: stat_bin : Removed 67 rows containing non-finite values.
    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/layer.py:401: PlotnineWarning: geom_histogram : Removed 20 rows containing missing values.
    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.


    Saving plot to results/bind_expr_filters/Omicron_BA2/count-score-histogram.pdf


    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/layer.py:324: PlotnineWarning: stat_bin : Removed 67 rows containing non-finite values.
    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/layer.py:401: PlotnineWarning: geom_histogram : Removed 20 rows containing missing values.



    
![png](bind_expr_filters_Omicron_BA2_files/bind_expr_filters_Omicron_BA2_14_3.png)
    


Things I want to know:
1. Mutations that have **any** counts in nature but are missing scores
2. Mutations that have appreciable counts (>=50) in nature but very low scores
3. The scores corresponding to the 95th percentile of all mutations occurring >= 50x in nature
4. The scores of mutations to disulfide bonds


```python
print('Here are the naturally occurring mutations that are missing scores from BA2 DMS')
display(HTML(dms_scores
             .query('count >= 1')
             .query('score.isnull()', engine='python')
             [['wildtype','mutation', 'count', 'n_countries', 'frequency', 'score']]
             .drop_duplicates()
             .to_html(index=False)
            )
       )
```

    Here are the naturally occurring mutations that are missing scores from BA2 DMS



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
    <tr>
      <td>F</td>
      <td>F392C</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>5.178956e-07</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>F</td>
      <td>F392L</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>3.107374e-06</td>
      <td>NaN</td>
    </tr>
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
      <td>Y</td>
      <td>Y501S</td>
      <td>501</td>
      <td>S</td>
      <td>94.0</td>
      <td>14.0</td>
      <td>0.000049</td>
      <td>delta_bind</td>
      <td>-2.92365</td>
      <td>&gt;=50</td>
    </tr>
    <tr>
      <td>A</td>
      <td>A419S</td>
      <td>419</td>
      <td>S</td>
      <td>313.0</td>
      <td>35.0</td>
      <td>0.000162</td>
      <td>delta_bind</td>
      <td>-2.66865</td>
      <td>&gt;=50</td>
    </tr>
  </tbody>
</table>


    Mutations with >=50 GISAID counts but with expr score < -0.95489



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
      <td>A411V</td>
      <td>411</td>
      <td>V</td>
      <td>53.0</td>
      <td>12.0</td>
      <td>0.000027</td>
      <td>delta_expr</td>
      <td>-1.32064</td>
      <td>&gt;=50</td>
    </tr>
    <tr>
      <td>P</td>
      <td>P521L</td>
      <td>521</td>
      <td>L</td>
      <td>50.0</td>
      <td>13.0</td>
      <td>0.000026</td>
      <td>delta_expr</td>
      <td>-1.27870</td>
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
      <td>-1.22493</td>
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
      <td>-1.22294</td>
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
      <td>-1.20472</td>
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
      <td>-1.18577</td>
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
      <td>-1.16370</td>
      <td>&gt;=50</td>
    </tr>
    <tr>
      <td>S</td>
      <td>S494L</td>
      <td>494</td>
      <td>L</td>
      <td>893.0</td>
      <td>49.0</td>
      <td>0.000462</td>
      <td>delta_expr</td>
      <td>-1.14887</td>
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
      <td>-1.11472</td>
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
      <td>-1.08799</td>
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
      <td>-1.07678</td>
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
      <td>-1.07272</td>
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
      <td>-1.06422</td>
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
      <td>-1.05082</td>
      <td>&gt;=50</td>
    </tr>
    <tr>
      <td>S</td>
      <td>S408I</td>
      <td>408</td>
      <td>I</td>
      <td>985.0</td>
      <td>31.0</td>
      <td>0.000510</td>
      <td>delta_expr</td>
      <td>-1.04031</td>
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
      <td>-1.02789</td>
      <td>&gt;=50</td>
    </tr>
    <tr>
      <td>A</td>
      <td>A372P</td>
      <td>372</td>
      <td>P</td>
      <td>67.0</td>
      <td>4.0</td>
      <td>0.000035</td>
      <td>delta_expr</td>
      <td>-0.98000</td>
      <td>&gt;=50</td>
    </tr>
    <tr>
      <td>A</td>
      <td>A344T</td>
      <td>344</td>
      <td>T</td>
      <td>144.0</td>
      <td>11.0</td>
      <td>0.000075</td>
      <td>delta_expr</td>
      <td>-0.97752</td>
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
      <td>-0.97070</td>
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
     xlab('WH1 RBD DMS score')
     )

fig = p.draw()

plotfile = os.path.join(resultsdir, f"disulfide-histogram.pdf")
print(f"Saving plot to {plotfile}")
p.save(plotfile, verbose=False)
```

    Here are the scores for mutations to disulfide bonds:


    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.


    Saving plot to results/bind_expr_filters/Omicron_BA2/disulfide-histogram.pdf


    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
    /fh/fast/bloom_j/computational_notebooks/tstarr/2022/SARS-CoV-2-RBD_Omicron_MAP_Overbaugh/env/lib/python3.8/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.



    
![png](bind_expr_filters_Omicron_BA2_files/bind_expr_filters_Omicron_BA2_18_4.png)
    


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
      <td>-1.888313</td>
      <td>-1.234609</td>
    </tr>
    <tr>
      <td>50</td>
      <td>2.5</td>
      <td>-1.446324</td>
      <td>-1.194298</td>
    </tr>
    <tr>
      <td>50</td>
      <td>5.0</td>
      <td>-0.903065</td>
      <td>-1.086869</td>
    </tr>
    <tr>
      <td>50</td>
      <td>10.0</td>
      <td>-0.740254</td>
      <td>-0.962838</td>
    </tr>
    <tr>
      <td>50</td>
      <td>25.0</td>
      <td>-0.435885</td>
      <td>-0.545845</td>
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
      <td>-0.08339</td>
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
      <td>-0.61624</td>
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
      <td>-0.14670</td>
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
      <td>-0.14146</td>
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
      <td>-0.53604</td>
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

print(f'BA2 SSM mutations that \npass bind: {n_bind} \npass expr: {n_expr} \npass both: {n_both} \npass both and not disulfide: {n_both_notC}')
print(f'Pass bind, expr, not disulfide, and not WT: {n_both_notC_notWT}')

print(f'Total number of possible mutations to non-disulfide sites: {total_muts_notC}')
```

    BA2 SSM mutations that 
    pass bind: 3375 
    pass expr: 2385 
    pass both: 2271 
    pass both and not disulfide: 2217
    Pass bind, expr, not disulfide, and not WT: 2024
    Total number of possible mutations to non-disulfide sites: 3649



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
    98.9
    This percentage of all variants seen >=50x in GISAID are retained by the expression filter of -0.95489
    89.6



```python

```
