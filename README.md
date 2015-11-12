# STOP
Scalable Transductive Learning over the Product Graph

## Usage
The program takes 4 files as its input:

A sparse graph G on the left and a sparse graph H on the right in the following format
```
vertexIn_G anotherVertexIn_G edgeStrength
```
Cross-graph links for training and testing
```
vertexIn_G vertexIn_H linkStrength
```
The vertex index can be any string. There's no need to convert the indices to consecutive integers.

E.g.,
G could be the social network among users; H could be a similarity graph of movies induced from their genres. In this case, cross-graph links may correspond to user-movie ratings.
  
By default, the program reads configurations specified in `*.ini`. The configuration file should be self-explanatory. Here is a sample pipeline for execution and evaluation:
```
make -j8 && ./train config/cmu.ini && python eval.py data/cmu/link.test.txt /tmp/predict.txt
```

## Author
Hanxiao Liu, Carnegie Mellon University
