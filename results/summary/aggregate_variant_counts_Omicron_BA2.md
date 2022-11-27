# Aggregate variant counts for all samples
Separate `Snakemake` rules count the observations of each variant in each sample from the Illumina barcode sequencing.
This Python Jupyter notebook aggregates all of this counts, and then adds them to a codon variant table.

## Set up analysis
### Import Python modules.
Use [plotnine](https://plotnine.readthedocs.io/en/stable/) for ggplot2-like plotting.

The analysis relies heavily on the Bloom lab's [dms_variants](https://jbloomlab.github.io/dms_variants) package:


```python
import glob
import itertools
import math
import os
import warnings

import Bio.SeqIO

import dms_variants.codonvarianttable
from dms_variants.constants import CBPALETTE
import dms_variants.utils
import dms_variants.plotnine_themes

from IPython.display import display, HTML

import pandas as pd

from plotnine import *

import yaml
```

Set [plotnine](https://plotnine.readthedocs.io/en/stable/) theme to the gray-grid one defined in `dms_variants`:


```python
theme_set(dms_variants.plotnine_themes.theme_graygrid())
```

Versions of key software:


```python
print(f"Using dms_variants version {dms_variants.__version__}")
```

    Using dms_variants version 0.8.10


Ignore warnings that clutter output:


```python
warnings.simplefilter('ignore')
```

Read the configuration file:


```python
with open('config.yaml') as f:
    config = yaml.safe_load(f)
```

Make output directory if needed:


```python
os.makedirs(config['counts_dir_Omicron_BA2'], exist_ok=True)
```

## Initialize codon variant table
Initialize the [CodonVariantTable](https://jbloomlab.github.io/dms_variants/dms_variants.codonvarianttable.html#dms_variants.codonvarianttable.CodonVariantTable) using the wildtype gene sequence and the CSV file with the table of variants:


```python
wt_seqrecord = Bio.SeqIO.read(config['wildtype_sequence_Omicron_BA2'], 'fasta')
geneseq = str(wt_seqrecord.seq)
primary_target = wt_seqrecord.name
print(f"Read sequence of {len(geneseq)} nt for {primary_target} from {config['wildtype_sequence_Omicron_BA2']}")
      
print(f"Initializing CodonVariantTable from gene sequence and {config['bc_variant_lookup_Omicron_BA2']}")
      
variants = dms_variants.codonvarianttable.CodonVariantTable(
                geneseq=geneseq,
                barcode_variant_file=config['bc_variant_lookup_Omicron_BA2'],
                substitutions_are_codon=True,
                substitutions_col='codon_substitutions',
                primary_target=primary_target)
```

    Read sequence of 603 nt for Omicron_BA2 from data/wildtype_sequence_Omicron_BA2.fasta
    Initializing CodonVariantTable from gene sequence and data/codon_variant_table_Omicron_BA2.csv


## Read barcode counts / fates
Read data frame with list of all samples (barcode runs):


```python
print(f"Reading list of barcode runs from {config['barcode_runs_Omicron_BA2']}")

barcode_runs = (pd.read_csv(config['barcode_runs_Omicron_BA2'])
                .assign(sample_lib=lambda x: x['sample'] + '_' + x['library'],
                        counts_file=lambda x: config['counts_dir_Omicron_BA2'] + '/' + x['sample_lib'] + '_counts.csv',
                        fates_file=lambda x: config['counts_dir_Omicron_BA2'] + '/' + x['sample_lib'] + '_fates.csv',
                        )
                .drop(columns='R1')  # don't need this column, and very large
                )

assert all(map(os.path.isfile, barcode_runs['counts_file'])), 'missing some counts files'
assert all(map(os.path.isfile, barcode_runs['fates_file'])), 'missing some fates files'

display(HTML(barcode_runs.to_html(index=False)))
```

    Reading list of barcode runs from data/barcode_runs_Omicron_BA2.csv



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>date</th>
      <th>experiment</th>
      <th>target</th>
      <th>library</th>
      <th>antibody</th>
      <th>concentration</th>
      <th>sort_bin</th>
      <th>selection</th>
      <th>sample</th>
      <th>experiment_type</th>
      <th>number_cells</th>
      <th>frac_escape</th>
      <th>sample_lib</th>
      <th>counts_file</th>
      <th>fates_file</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>220816</td>
      <td>exptREF</td>
      <td>Omicron_BA2</td>
      <td>lib24</td>
      <td>none</td>
      <td>0</td>
      <td>ref</td>
      <td>reference</td>
      <td>exptREF-none-0-ref</td>
      <td>ab_selection</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>exptREF-none-0-ref_lib24</td>
      <td>results/counts/Omicron_BA2/exptREF-none-0-ref_lib24_counts.csv</td>
      <td>results/counts/Omicron_BA2/exptREF-none-0-ref_lib24_fates.csv</td>
    </tr>
    <tr>
      <td>220816</td>
      <td>exptREF</td>
      <td>Omicron_BA2</td>
      <td>lib25</td>
      <td>none</td>
      <td>0</td>
      <td>ref</td>
      <td>reference</td>
      <td>exptREF-none-0-ref</td>
      <td>ab_selection</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>exptREF-none-0-ref_lib25</td>
      <td>results/counts/Omicron_BA2/exptREF-none-0-ref_lib25_counts.csv</td>
      <td>results/counts/Omicron_BA2/exptREF-none-0-ref_lib25_fates.csv</td>
    </tr>
    <tr>
      <td>220816</td>
      <td>expt10</td>
      <td>Omicron_BA2</td>
      <td>lib24</td>
      <td>268C3</td>
      <td>180</td>
      <td>abneg</td>
      <td>escape</td>
      <td>expt10-268C3-180-abneg</td>
      <td>NaN</td>
      <td>467769.0</td>
      <td>0.136</td>
      <td>expt10-268C3-180-abneg_lib24</td>
      <td>results/counts/Omicron_BA2/expt10-268C3-180-abneg_lib24_counts.csv</td>
      <td>results/counts/Omicron_BA2/expt10-268C3-180-abneg_lib24_fates.csv</td>
    </tr>
    <tr>
      <td>220816</td>
      <td>expt10</td>
      <td>Omicron_BA2</td>
      <td>lib25</td>
      <td>268C3</td>
      <td>180</td>
      <td>abneg</td>
      <td>escape</td>
      <td>expt10-268C3-180-abneg</td>
      <td>NaN</td>
      <td>459335.0</td>
      <td>0.135</td>
      <td>expt10-268C3-180-abneg_lib25</td>
      <td>results/counts/Omicron_BA2/expt10-268C3-180-abneg_lib25_counts.csv</td>
      <td>results/counts/Omicron_BA2/expt10-268C3-180-abneg_lib25_fates.csv</td>
    </tr>
    <tr>
      <td>220816</td>
      <td>expt12</td>
      <td>Omicron_BA2</td>
      <td>lib24</td>
      <td>268C61</td>
      <td>98</td>
      <td>abneg</td>
      <td>escape</td>
      <td>expt12-268C61-98-abneg</td>
      <td>NaN</td>
      <td>502055.0</td>
      <td>0.146</td>
      <td>expt12-268C61-98-abneg_lib24</td>
      <td>results/counts/Omicron_BA2/expt12-268C61-98-abneg_lib24_counts.csv</td>
      <td>results/counts/Omicron_BA2/expt12-268C61-98-abneg_lib24_fates.csv</td>
    </tr>
    <tr>
      <td>220816</td>
      <td>expt12</td>
      <td>Omicron_BA2</td>
      <td>lib25</td>
      <td>268C61</td>
      <td>98</td>
      <td>abneg</td>
      <td>escape</td>
      <td>expt12-268C61-98-abneg</td>
      <td>NaN</td>
      <td>483520.0</td>
      <td>0.138</td>
      <td>expt12-268C61-98-abneg_lib25</td>
      <td>results/counts/Omicron_BA2/expt12-268C61-98-abneg_lib25_counts.csv</td>
      <td>results/counts/Omicron_BA2/expt12-268C61-98-abneg_lib25_fates.csv</td>
    </tr>
  </tbody>
</table>


Confirm sample / library combinations unique:


```python
assert len(barcode_runs) == len(barcode_runs.groupby(['sample', 'library']))
```

Make sure the the libraries for which we have barcode runs are all in our variant table:


```python
unknown_libs = set(barcode_runs['library']) - set(variants.libraries)
if unknown_libs:
    raise ValueError(f"Libraries with barcode runs not in variant table: {unknown_libs}")
```

Now concatenate the barcode counts and fates for each sample:


```python
counts = pd.concat([pd.read_csv(f) for f in barcode_runs['counts_file']],
                   sort=False,
                   ignore_index=True)

print('First few lines of counts data frame:')
display(HTML(counts.head().to_html(index=False)))

fates = pd.concat([pd.read_csv(f) for f in barcode_runs['fates_file']],
                  sort=False,
                  ignore_index=True)

print('First few lines of fates data frame:')
display(HTML(fates.head().to_html(index=False)))
```

    First few lines of counts data frame:



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>barcode</th>
      <th>count</th>
      <th>library</th>
      <th>sample</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>GTGTGGCCATATGAGA</td>
      <td>9179</td>
      <td>lib24</td>
      <td>exptREF-none-0-ref</td>
    </tr>
    <tr>
      <td>TATAAAGCACATGCAG</td>
      <td>8658</td>
      <td>lib24</td>
      <td>exptREF-none-0-ref</td>
    </tr>
    <tr>
      <td>AGTGGAGCCGCCAAAC</td>
      <td>8128</td>
      <td>lib24</td>
      <td>exptREF-none-0-ref</td>
    </tr>
    <tr>
      <td>GATGCCGGTGAGATAG</td>
      <td>8001</td>
      <td>lib24</td>
      <td>exptREF-none-0-ref</td>
    </tr>
    <tr>
      <td>GTACAGCATCTGAGTT</td>
      <td>7858</td>
      <td>lib24</td>
      <td>exptREF-none-0-ref</td>
    </tr>
  </tbody>
</table>


    First few lines of fates data frame:



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>fate</th>
      <th>count</th>
      <th>library</th>
      <th>sample</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>valid barcode</td>
      <td>69708086</td>
      <td>lib24</td>
      <td>exptREF-none-0-ref</td>
    </tr>
    <tr>
      <td>low quality barcode</td>
      <td>13378411</td>
      <td>lib24</td>
      <td>exptREF-none-0-ref</td>
    </tr>
    <tr>
      <td>invalid barcode</td>
      <td>9075636</td>
      <td>lib24</td>
      <td>exptREF-none-0-ref</td>
    </tr>
    <tr>
      <td>unparseable barcode</td>
      <td>1392028</td>
      <td>lib24</td>
      <td>exptREF-none-0-ref</td>
    </tr>
    <tr>
      <td>failed chastity filter</td>
      <td>0</td>
      <td>lib24</td>
      <td>exptREF-none-0-ref</td>
    </tr>
  </tbody>
</table>


## Examine fates of parsed barcodes
First, we'll analyze the "fates" of the parsed barcodes.
These fates represent what happened to each Illumina read we parsed:
 - Did the barcode read fail the Illumina chastity filter?
 - Was the barcode *unparseable* (i.e., the read didn't appear to be a valid barcode based on flanking regions)?
 - Was the barcode sequence too *low quality* based on the Illumina quality scores?
 - Was the barcode parseable but *invalid* (i.e., not in our list of variant-associated barcodes in the codon variant table)?
 - Was the barcode *valid*, and so will be added to variant counts.
 
First, we just write a CSV file with all the barcode fates:


```python
fatesfile = os.path.join(config['counts_dir_Omicron_BA2'], 'barcode_fates.csv')
print(f"Writing barcode fates to {fatesfile}")
fates.to_csv(fatesfile, index=False)
```

    Writing barcode fates to results/counts/Omicron_BA2/barcode_fates.csv


Next, we tabulate the barcode fates in wide format:


```python
display(HTML(fates
             .pivot_table(columns='fate',
                          values='count',
                          index=['sample', 'library'])
             .applymap('{:.1e}'.format)  # scientific notation
             .to_html()
             ))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fate</th>
      <th>failed chastity filter</th>
      <th>invalid barcode</th>
      <th>low quality barcode</th>
      <th>unparseable barcode</th>
      <th>valid barcode</th>
    </tr>
    <tr>
      <th>sample</th>
      <th>library</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">expt10-268C3-180-abneg</th>
      <th>lib24</th>
      <td>0.0e+00</td>
      <td>4.1e+05</td>
      <td>5.6e+05</td>
      <td>5.7e+04</td>
      <td>2.9e+06</td>
    </tr>
    <tr>
      <th>lib25</th>
      <td>0.0e+00</td>
      <td>3.4e+05</td>
      <td>4.9e+05</td>
      <td>5.0e+04</td>
      <td>2.5e+06</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">expt12-268C61-98-abneg</th>
      <th>lib24</th>
      <td>0.0e+00</td>
      <td>3.9e+05</td>
      <td>5.3e+05</td>
      <td>5.7e+04</td>
      <td>2.8e+06</td>
    </tr>
    <tr>
      <th>lib25</th>
      <td>0.0e+00</td>
      <td>3.5e+05</td>
      <td>5.0e+05</td>
      <td>5.4e+04</td>
      <td>2.6e+06</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">exptREF-none-0-ref</th>
      <th>lib24</th>
      <td>0.0e+00</td>
      <td>9.1e+06</td>
      <td>1.3e+07</td>
      <td>1.4e+06</td>
      <td>7.0e+07</td>
    </tr>
    <tr>
      <th>lib25</th>
      <td>0.0e+00</td>
      <td>4.3e+06</td>
      <td>6.3e+06</td>
      <td>6.8e+05</td>
      <td>3.3e+07</td>
    </tr>
  </tbody>
</table>


Now we plot the barcode-read fates for each library / sample, showing the bars for valid barcodes in orange and the others in gray.
We see that the largest fraction of barcode reads correspond to valid barcodes, and most of the others are invalid barcodes (probably because the map to variants that aren't present in our variant table since we didn't associate all variants with barcodes). The exception to this is lib2 Titeseq_03_bin3; the PCR for this sample in the original sequencing run failed, so we followed it up with a single MiSeq lane. We did not filter out the PhiX reads from this data before parsing, so these PhiX reads will deflate the fraction of valid barcode reads as expected, but does not indicate any problems.


```python
ncol = 4
nfacets = len(fates.groupby(['sample', 'library']))

barcode_fate_plot = (
    ggplot(
        fates
        .assign(sample=lambda x: pd.Categorical(x['sample'],
                                                x['sample'].unique(),
                                                ordered=True),
                fate=lambda x: pd.Categorical(x['fate'],
                                              x['fate'].unique(),
                                              ordered=True),
                is_valid=lambda x: x['fate'] == 'valid barcode'
                ), 
        aes('fate', 'count', fill='is_valid')) +
    geom_bar(stat='identity') +
    facet_wrap('~ sample + library', ncol=ncol) +
    scale_fill_manual(CBPALETTE, guide=False) +
    theme(figure_size=(3.25 * ncol, 2 * math.ceil(nfacets / ncol)),
          axis_text_x=element_text(angle=90),
          panel_grid_major_x=element_blank()
          ) +
    scale_y_continuous(labels=dms_variants.utils.latex_sci_not,
                       name='number of reads')
    )

_ = barcode_fate_plot.draw()
```


    
![png](aggregate_variant_counts_Omicron_BA2_files/aggregate_variant_counts_Omicron_BA2_28_0.png)
    


## Add barcode counts to variant table
Now we use the [CodonVariantTable.add_sample_counts_df](https://jbloomlab.github.io/dms_variants/dms_variants.codonvarianttable.html#dms_variants.codonvarianttable.CodonVariantTable.add_sample_counts_df) method to add the barcode counts to the variant table:


```python
variants.add_sample_counts_df(counts)
```

The variant table now has a `variant_count_df` attribute that gives a data frame of all the variant counts.
Here are the first few lines:


```python
display(HTML(variants.variant_count_df.head().to_html(index=False)))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>target</th>
      <th>library</th>
      <th>sample</th>
      <th>barcode</th>
      <th>count</th>
      <th>variant_call_support</th>
      <th>codon_substitutions</th>
      <th>aa_substitutions</th>
      <th>n_codon_substitutions</th>
      <th>n_aa_substitutions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Omicron_BA2</td>
      <td>lib24</td>
      <td>exptREF-none-0-ref</td>
      <td>GTGTGGCCATATGAGA</td>
      <td>9179</td>
      <td>108</td>
      <td>TTG122ATG</td>
      <td>L122M</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>Omicron_BA2</td>
      <td>lib24</td>
      <td>exptREF-none-0-ref</td>
      <td>TATAAAGCACATGCAG</td>
      <td>8658</td>
      <td>103</td>
      <td>AAT13ATG</td>
      <td>N13M</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>Omicron_BA2</td>
      <td>lib24</td>
      <td>exptREF-none-0-ref</td>
      <td>AGTGGAGCCGCCAAAC</td>
      <td>8128</td>
      <td>77</td>
      <td>GTT32ATT</td>
      <td>V32I</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>Omicron_BA2</td>
      <td>lib24</td>
      <td>exptREF-none-0-ref</td>
      <td>GATGCCGGTGAGATAG</td>
      <td>8001</td>
      <td>95</td>
      <td>GTT115ATG</td>
      <td>V115M</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>Omicron_BA2</td>
      <td>lib24</td>
      <td>exptREF-none-0-ref</td>
      <td>GTACAGCATCTGAGTT</td>
      <td>7858</td>
      <td>86</td>
      <td>AAG198GTT</td>
      <td>K198V</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>


Write the variant counts data frame to a CSV file.
It can then be used to re-initialize a [CodonVariantTable](https://jbloomlab.github.io/dms_variants/dms_variants.codonvarianttable.html#dms_variants.codonvarianttable.CodonVariantTable) via its [from_variant_count_df](https://jbloomlab.github.io/dms_variants/dms_variants.codonvarianttable.html#dms_variants.codonvarianttable.CodonVariantTable.from_variant_count_df) method:


```python
print(f"Writing variant counts to {config['variant_counts_Omicron_BA2']}")
variants.variant_count_df.to_csv(config['variant_counts_Omicron_BA2'], index=False, compression='gzip')
```

    Writing variant counts to results/counts/Omicron_BA2/variant_counts.csv.gz


The [CodonVariantTable](https://jbloomlab.github.io/dms_variants/dms_variants.codonvarianttable.html#dms_variants.codonvarianttable.CodonVariantTable) has lots of nice functions that can be used to analyze the counts it contains.
However, we do that in the next notebook so we don't have to re-run this entire (rather computationally intensive) notebook every time we want to analyze a new aspect of the counts.
