# TOP++
Large-scale Transductive Learning over Product Graphs
```
Options:
  -G FILE, --entityGraphG=FILE      entity graph G on the left
  -H FILE, --entityGraphH=FILE      entity graph H on the right
  -T FILE, --trainingLinks=FILE     cross-graph links for training
  -V FILE, --validationLinks=FILE   cross-graph links for validation
  -O FILE, --predictions=FILE       where to store predictions

  -k INT, --dimF=INT                inner dimension of F (50)
  -p INT, --dimG=INT                inner dimension of G (100)
  -q INT, --dimH=INT                inner dimension of H (100)
  -C DOUBLE, --C=DOUBLE             C*loss_func(F) + regularization(F) (1)

  --algorithm=[top|pmf]             algorithm (top)
  --convergence=DOUBLE              desired convergence rate (0.001)
  --decay=DOUBLE                    decay factor for infinite ramdom walk (1)
  --alpha=DOUBLE                    backtracking parameter: \alpha (0.5)
  --beta=DOUBLE                     backtracking parameter: \beta (0.5)
  --PCGTolerance=DOUBLE             PCG tolerance (1e-05)
  --PCGMaxIter=INT                  max PCG iterations (50)
  --eta0=DOUBLE                     PMF learning rate (0.001)
  --maxThreads=INT                  max number of training threads (4)
  --inferDump=FILE                  when specified, dump the induced top
                                    [inferTop] entities in H for each entity in G
  --inferTop=INT                    see above (10)
```

## Input Format
The four input files are in the following 3-column format
```
STRING STRING DOUBLE
source target linkStrength(source, target)
```
Note there is no need to convert the words into integer-valued indices.

## Example: Semi-supervised Word Translation

### Description
The dataset is located at `data/bilingual`,
where `cn.graph.txt` is a 50-NN similarity graph of Chinese word induced from an external corpus in via word embeddings. Similarly, `en.graph.txt` is the graph for English words.

With info provided in the two monolingual graphs,
our goal is to correctly rank the cross-language word translations in `validationLinks.txt` based on a set of seed translations in `trainingLinks.txt` obtained from a Chinese-English dictionary.
  
The code can be compiled with `make -j8` via g++

Training and predicting
```
./top \
-G data/bilingual/cn.graph.txt \
-H data/bilingual/en.graph.txt \
-T data/bilingual/dict.trn.txt \
-V data/bilingual/dict.val.txt \
-O /tmp/predictions.txt \
-k 100 -p 500 -q 500 -C 5 --decay=5
```

Evaluating on the validation set with Mean Average Prevision (MAP) @1-10
```
python eval.py data/bilingual/dict.val.txt /tmp/predictions.txt
```

###Evaluation
Performance over the test set

| Algorithm            | RMSE   | MAE    |
|----------------------|--------|--------|
| TOP++                | **0.0935** | **0.0215** |
| Graph Regularized MF | 0.1010 | 0.0229 |
| Probabilistic MF     | 0.1113 | 0.0339 |

| MAP                  | @1    | @2    | @3    | @4    | @5    | @6    | @7    | @8    | @9    | @10   |
|----------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| TOP++                | **0.464** | **0.561** | **0.593** | **0.610** | **0.617** | **0.622** | **0.625** | **0.628** | **0.630** | **0.631** |
| Graph Regularized MF | 0.172 | 0.206 | 0.221 | 0.229 | 0.235 | 0.239 | 0.243 | 0.247 | 0.250 | 0.253 |
| Probabilistic MF     | 0.127 | 0.155 | 0.166 | 0.173 | 0.179 | 0.183 | 0.186 | 0.189 | 0.191 | 0.193 |

Sample output of system-induced translations with `--inferDump=FILE` and `inferTop=5` options
```
统领      chief          0.716054               大洲      continent      0.385417
统领      manager        0.650474               大洲      country        0.350127
统领      director       0.621973               大洲      region         0.340655
统领      coordinator    0.618977               大洲      world          0.323181
统领      adviser        0.617633               大洲      regional       0.318236
园艺      agriculture    0.242977               出名      great          0.288545
园艺      property       0.229218               出名      difficult      0.280768
园艺      flower         0.228028               出名      good           0.26863
园艺      farm           0.219118               出名      rare           0.267517
园艺      farmland       0.207904               出名      remarkable     0.262598
模具      matrix         0.549939               急迫      urgent         0.851337
模具      mold           0.54591                急迫      important      0.745108
模具      equipment      0.493759               急迫      pressing       0.725116
模具      goods          0.460778               急迫      critical       0.675325
模具      component      0.450376               急迫      essential      0.66111
水运      thoroughfare   0.350764               信息网    access         0.235622
水运      road           0.340089               信息网    digital        0.224737
水运      traffic        0.32164                信息网    website        0.220754
水运      route          0.32011                信息网    system         0.214955
水运      waterway       0.314854               信息网    broadband      0.213725
侧重      style          0.122329               别有用心  evil           0.183328
侧重      benefit        0.117328               别有用心  cruel          0.16004
侧重      orientation    0.115426               别有用心  rumor          0.154512
侧重      profit         0.11258                别有用心  idea           0.152945
侧重      similar        0.109831               别有用心  ridiculous     0.146974
诗句      verse          0.38904                不切实际  unnecessary    0.317531
诗句      words          0.344987               不切实际  inappropriate  0.3153
诗句      happy          0.33975                不切实际  ridiculous     0.313933
诗句      sincere        0.320716               不切实际  appropriate    0.309792
诗句      poem           0.318363               不切实际  excessive      0.292981
```

## Author
Copyright (c) 2015 Hanxiao Liu, Carnegie Mellon University
