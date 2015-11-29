# STOP
Transductive Learning over Product Graphs
```
Options:
  -G FILE, --entityGraphG=FILE      entity graph G on the left
  -H FILE, --entityGraphH=FILE      entity graph H on the right
  -T FILE, --trainingLinks=FILE     cross-graph links for training
  -V FILE, --validationLinks=FILE   cross-graph links for validation
  -O FILE, --predictions=FILE       predictions

  -k INT, --dimensionF=INT          inner dimension of F (50)
  -p INT, --dimensionG=INT          inner dimension of G (100)
  -q INT, --dimensionH=INT          inner dimension of H (100)
  -C DOUBLE, --C=DOUBLE             C*loss_fun(F) + regularization(F) (1)
  -e DOUBLE, --convergence=DOUBLE   desired convergence rate (0.001)
  --algorithm=[top|pmf]             algorithm (top)
  --decay=DOUBLE                    decay factor for infinite ramdom walk (1)
  --alpha=DOUBLE                    backtracking parameter: \alpha (0.5)
  --beta=DOUBLE                     backtracking parameter: \beta (0.5)
  --PCGTolerance=DOUBLE             PCG tolerance (0.001)
  --PCGMaxIter=INT                  max PCG iterations (50)
  --eta0=DOUBLE                     PMF learning rate (0.001)
```

## Input Format
The four input files are in the following 3-column format
```
STRING STRING DOUBLE
source target linkStrength(source, target)
```
There is no need to convert the words into integer-valued indices.

## Example: Semi-supervised Word Translation

The dataset is located at `data/bilingual`,
where `cn.graph.txt` is a 50-NN similarity graph of Chinese word induced from an external corpus in via word embeddings. Similarly, `en.graph.txt` is the graph for English words.

With info provided in the two monolingual graphs,
our goal is to correctly rank the cross-language word translations in `validationLinks.txt` based on a set of seed translations in `trainingLinks.txt` obtained from a Chinese-English dictionary.
  
Compling: `make -j8`

Training and predicting
```
./top \
-G data/bilingual/cn.graph.txt \
-H data/bilingual/en.graph.txt \
-T data/bilingual/dict.trn.txt \
-V data/bilingual/dict.val.txt \
-O /tmp/predictions.txt \
-k 100 -p 500 -q 500 -C 5 --decay=2
```

Evaluating on the validation set with Mean Average Prevision (MAP) @1-10
```
python eval.py data/bilingual/dict.val.txt /tmp/predictions.txt
```

## Sample Output
TBA

## Author
Hanxiao Liu, Carnegie Mellon University
