/*
 * Large-scale Transduction over the Product Graph
 * @author Hanxiao Liu, Carnegie Mellon University
 */
using namespace std;

#include <assert.h>
#include "Top.hh"
#include "OptionParser.h"

using namespace optparse;

int main(int argc, char *argv[]) {

    OptionParser parser = OptionParser()
    .version("1.0\nCopyright (C) 2015 Hanxiao Liu\n")
    .usage(SUPPRESS_USAGE)
    .description("(T)ransductive Learning (o)ver (P)roduct Graphs\n\
           _                _                _      \n\
          /\\ \\             /\\ \\             /\\ \\    \n\
          \\_\\ \\           /  \\ \\           /  \\ \\   \n\
          /\\__ \\         / /\\ \\ \\         / /\\ \\ \\  \n\
         / /_ \\ \\       / / /\\ \\ \\       / / /\\ \\_\\ \n\
        / / /\\ \\ \\     / / /  \\ \\_\\     / / /_/ / / \n\
       / / /  \\/_/    / / /   / / /    / / /__\\/ /  \n\
      / / /          / / /   / / /    / / /_____/   \n\
     / / /          / / /___/ / /    / / /          \n\
    /_/ /          / / /____\\/ /    / / /           \n\
    \\_\\/           \\/_________/     \\/_/            ");

    parser.add_option("-G", "--entityGraphG") .metavar("FILE") .help("entity graph G on the left");
    parser.add_option("-H", "--entityGraphH") .metavar("FILE") .help("entity graph H on the right");
    parser.add_option("-T", "--trainingLinks") .metavar("FILE") .help("cross-graph links for training");
    parser.add_option("-V", "--validationLinks") .metavar("FILE") .help("cross-graph links for validation");
    parser.add_option("-O", "--predictions") .metavar("FILE") .help("predictions\n");
    
    parser.add_option("-k", "--dimensionF") .type("int") .set_default(50) .help("inner dimension of F (%default)");
    parser.add_option("-p", "--dimensionG") .type("int") .set_default(100) .help("inner dimension of G (%default)");
    parser.add_option("-q", "--dimensionH") .type("int") .set_default(100) .help("inner dimension of H (%default)");
    parser.add_option("-C", "--C") .type("double") .set_default(1.0) .help("C*loss_fun(F) + regularization(F) (%default)");

    char const* const algs[] = { "top", "pmf" };
    parser.add_option("-e", "--convergence") .type("double") .set_default(1e-3) .help("desired convergence rate (%default)");

    parser.add_option("--algorithm") .metavar("[top|pmf]") .choices(&algs[0], &algs[2]) .set_default("top") .help("algorithm (%default)");
    parser.add_option("--decay") .type("double") .set_default(1.0) .help("decay factor for infinite ramdom walk (%default)");
    parser.add_option("--alpha") .type("double") .set_default(0.5) .help("backtracking parameter: \\alpha (%default)");
    parser.add_option("--beta") .type("double") .set_default(0.5) .help("backtracking parameter: \\beta (%default)");
    parser.add_option("--PCGTolerance") .type("double") .set_default(1e-5) .help("PCG tolerance (%default)");
    parser.add_option("--PCGMaxIter") .type("int") .set_default(50) .help("max PCG iterations (%default)");
    parser.add_option("--eta0") .type("double") .set_default(1e-3) .help("PMF learning rate (%default)");

    Values& options = parser.parse_args(argc, argv);

    const char* entityGraphG = (const char*) options.get("entityGraphG");
    const char* entityGraphH = (const char*) options.get("entityGraphH");
    const char* trainingLinks = (const char*) options.get("trainingLinks");
    const char* validationLinks = (const char*) options.get("validationLinks");
    const char* predictions = (const char*) options.get("predictions");

    if ( !(strlen(entityGraphG) && strlen(entityGraphH) && strlen(trainingLinks) && strlen(validationLinks) &&  strlen(predictions)) ) {
        parser.print_help();
        exit(1);
    }

    Config opt;
    opt.k_f = (unsigned) options.get("dimensionF");
    opt.k_g = (unsigned) options.get("dimensionG");
    opt.k_h = (unsigned) options.get("dimensionH");
    opt.decay = (double) options.get("decay");
    opt.C = (double) options.get("C");
    opt.alg = (const char*) options.get("algorithm");
    opt.tol = (double) options.get("convergence");
    opt.alpha = (double) options.get("alpha");
    opt.beta = (double) options.get("beta");
    opt.pcgTol = (double) options.get("PCGTolerance");
    opt.pcgIter = (double) options.get("PCGMaxIter");
    opt.eta0 = (double) options.get("eta0");

    /*load graph G on the left*/
    fprintf(stderr, "Loading G ... ");
    Entity g(entityGraphG);
    fprintf(stderr, "Nodes in Entity 1:\t%d\n", g.n);

    /*load graph H on the right*/
    fprintf(stderr, "Loading H ... ");
    Entity h(entityGraphH);
    fprintf(stderr, "Nodes in Entity 2:\t%d\n", h.n);

    /*load observed cross-graph links for training*/
    fprintf(stderr, "Loading training links ... ");
    Relation trn(trainingLinks, g, h);
    fprintf(stderr, "Edges for Training:\t%zd\n", trn.edges.size());

    /*load hold-out cross-graph links for testing*/
    fprintf(stderr, "Loading test links ... ");
    Relation tes(validationLinks, g, h);
    fprintf(stderr, "Edges for Testing:\t%zd\n", tes.edges.size());

    /*initialize the algorithm*/
    Top top(opt);

    /*training*/
    fprintf(stderr, "Training with %d threads ...\n", Eigen::nbThreads());
    assert(top.train(g, h, trn, tes));

    /*dump the predictions to file*/
    fprintf(stderr, "Predicting ...\n");
    Result result = top.predict(g, h, tes, predictions);
    fprintf(stderr, "MAE  = %f\n", result.mae);
    fprintf(stderr, "RMSE = %f\n", result.rmse);
}
