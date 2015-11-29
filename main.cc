/*
 * Large-scale Transduction over the Product Graph
 * @author Hanxiao Liu, Carnegie Mellon University
 */
using namespace std;

#include <assert.h>
#include <iomanip>
#include "Top.hh"
#include "OptionParser.h"

using namespace optparse;

#define COLOR_1 "\033[35m"
#define COLOR_2 "\033[36m"
#define RESET "\033[0m"
#define INFO_1(X, Y) cerr << X << COLOR_1 << Y << RESET << endl;
#define INFO_2(X, Y) cerr << left << setw(18) << X << COLOR_2 << Y << RESET << endl;

bool myComparator(const tuple<int, val>& l, const tuple<int, val>& r) { return get<1>(l) > get<1>(r); }

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
    parser.add_option("--inferDump") .metavar("FILE") .help("when specified, dump the highest scored [inferTop] entities in H for each entity in G");
    parser.add_option("--inferTop") .type("int") .set_default(10) .help("see above (%default)");

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
    INFO_1("G <-- ", entityGraphG) 
    Entity g(entityGraphG);

    /*load graph H on the right*/
    INFO_1("H <-- ", entityGraphH) 
    Entity h(entityGraphH);

    /*load observed cross-graph links for training*/
    INFO_1("T <-- ", trainingLinks) 
    Relation trn(trainingLinks, g, h);

    /*load hold-out cross-graph links for testing*/
    INFO_1("V <-- ", validationLinks)
    Relation tes(validationLinks, g, h);

    INFO_2("Entities in G: ", g.n)
    INFO_2("Entities in H: ", h.n)
    INFO_2("Training edges: ", trn.edges.size())
    INFO_2("Validation edges: ", trn.edges.size())
    INFO_2("CPU threads: ", Eigen::nbThreads())

    /*initialize the algorithm*/
    Top top(opt);

    /*training*/
    assert(top.train(g, h, trn, tes));

    /*dump the predictions to file*/
    INFO_1("Predictions --> ", predictions)
    Result result = top.predict(g, h, tes, predictions);
    INFO_2("MAE", result.mae)
    INFO_2("RMSE", result.rmse)

    const char* inferDump = (const char*) options.get("inferDump");

    if ( strlen(inferDump) ) {
        INFO_1("Induced Links --> ", inferDump)
        unsigned inferTop = (unsigned) options.get("inferTop");

        mat L = top.get_L();
        mat R = top.get_R();

        vector< tuple<int, val> > topPairs(inferTop);

        string gId;
        ofstream ofs(inferDump);
        if ( !ofs.fail() ) {
            mat f, scores;
            val threshold;
            for (unsigned i = 0; i < L.rows(); i++) {
                f = R * L.row(i).transpose();

                scores = f;
                nth_element(scores.data(), scores.data()+inferTop, scores.data()+scores.size(), greater<val>());
                threshold = scores(inferTop);

                unsigned k = 0;
                for (unsigned j = 0; j < R.rows(); j++) {
                    if ( f(j) > threshold )
                        topPairs[k++] = tuple<int, val> (j, f(j));
                    if ( k == inferTop ) break;
                }
                sort(topPairs.begin(), topPairs.begin()+k, myComparator); 
                gId = g.id_of.at(i);
                for (unsigned j = 0; j < k; j++)
                    ofs << gId << " " << h.id_of.at(get<0>(topPairs[j])) << " " << get<1>(topPairs[j]) << endl;
            }
            ofs.close();
        }
    }
}
