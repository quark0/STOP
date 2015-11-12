/*
 * Large-scale Transduction over the Product Graph
 * @author Hanxiao Liu, Carnegie Mellon University
 */
using namespace std;

#include <assert.h>
#include "Top.hh"
#include "SimpleIni.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s configuration.ini\n", argv[0]);
        exit(1);
    }
    /*load the configuration file *.ini*/
    CSimpleIniA ini(true, true, true);
    assert(ini.LoadFile(argv[1]) >= 0);

    /*load algorithmic settings from file*/
    Option opt;
    opt.k_f = ini.GetLongValue("Model", "k_f", 50);
    opt.k_g = ini.GetLongValue("Model", "k_g", 100);
    opt.k_h = ini.GetLongValue("Model", "k_h", 100);
    opt.decay = ini.GetDoubleValue("Model", "decay", 1);
    opt.C = ini.GetDoubleValue("Model", "C", 1e-3);
    opt.alg = ini.GetLongValue("Model", "alg", 1);
    opt.tol = ini.GetDoubleValue("Optimization", "tol", 1e-3);
    opt.alpha = ini.GetDoubleValue("Optimization", "alpha", 0.5);
    opt.beta = ini.GetDoubleValue("Optimization", "beta", 0.5);            
    opt.pcgTol = ini.GetDoubleValue("Optimization", "pcgTol", 1e-2);
    opt.pcgIter = ini.GetLongValue("Optimization", "pcgIter", 50);
    opt.eta0 = ini.GetDoubleValue("Optimization", "eta0", 1e-3);

    /*load graph G on the left*/
    fprintf(stderr, "Loading G ... ");
    Entity g(ini.GetValue("IO", "G", NULL));
    fprintf(stderr, "Nodes in Entity 1:\t%d\n", g.n);

    /*load graph H on the right*/
    fprintf(stderr, "Loading H ... ");
    Entity h(ini.GetValue("IO", "H", NULL));
    fprintf(stderr, "Nodes in Entity 2:\t%d\n", h.n);

    /*load observed cross-graph links for training*/
    fprintf(stderr, "Loading training links ... ");
    Relation trn(ini.GetValue("IO", "linksTrain", NULL), g, h);
    fprintf(stderr, "Edges for Training:\t%zd\n", trn.edges.size());

    /*load hold-out cross-graph links for testing*/
    fprintf(stderr, "Loading test links ... ");
    Relation tes(ini.GetValue("IO", "linksTest", NULL), g, h);
    fprintf(stderr, "Edges for Testing:\t%zd\n", tes.edges.size());

    /*initialize the algorithm*/
    Top top(opt);

    /*training*/
    fprintf(stderr, "Training with %d threads ...\n", Eigen::nbThreads());
    assert(top.train(g, h, trn, tes));

    /*dump the predictions to file*/
    fprintf(stderr, "Predicting ...\n");
    Result result = top.predict(g, h, tes, ini.GetValue("IO", "prediction", "prediction.txt"));
    fprintf(stderr, "MAE  = %f\n", result.mae);
    fprintf(stderr, "RMSE = %f\n", result.rmse);
}
