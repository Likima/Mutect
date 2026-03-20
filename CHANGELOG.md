# Changelog

## March 19, 2026

### Model Improvements:

1. added XGBoost to RF classification
2. tuned features such that importance is dispersed over different features rather than being dominated by 3
3. Scoring of single allele repeats scaled down -> fixes the misclassification of STR's with more detail to them.

## General Improvements:

1. Reimplemented sliding window approach that allows the model to classify STR's over multiple reads. Averages out STR probabilities across each window.
2. Added existing research into LRS STR classification through REDatlas and TRMotifAnnotator
3. Added common diseases associated with STR expansions. Model can now classify WHAT disesease is associated with an STR expansion.

## Tests:

1. Tested HG01122 ATXN10 on Chromosome 22 from 45790000 to 45800000 where it correctly outputted

```bash
Predictions: 1 STRs, 61 non-STRs
Average STR probability: 0.8853

Top 10 predicted STRs (highest probability):
  1. Prob=0.8853, Motif=(ATTCT) x 983 [Spinocerebellar ataxia 10]: CCATGTATTGCTAGTCTGTGAGGTTCCTTCAGGTTTGGAACTGTCTTTTATTAATCTTTA...
```