using namespace std;

#include <assert.h>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>

#include "Top.hh"
#include "OptionParser.h"

using namespace optparse;

#define COLOR_1 "\033[35m"
#define COLOR_2 "\033[36m"
#define RESET "\033[0m"
#define INFO_1(X, Y) cerr << X << COLOR_1 << Y << RESET << endl;
#define INFO_2(X, Y) cerr << left << setw(18) << X << COLOR_2 << Y << RESET << endl;

bool tuple_comparator(const tuple<int, val>& l, const tuple<int, val>& r) {
    return get<1>(l) > get<1>(r);
}

void dump_inference(const char* dump_file, unsigned numInfer, Top* top, const Entity& g, const Entity& h) {
    mat L = top -> get_L();
    mat R = top -> get_R();
    vector< tuple<int, val> > top_pairs(numInfer);

    string gid;
    ofstream ofs(dump_file);
    if ( !ofs.fail() ) {
        mat f, scores;
        val threshold;
        for (unsigned i = 0; i < L.rows(); i++) {
            f = R * L.row(i).transpose();
            // find cutting-point for the top-ranked components in f
            scores = f;
            nth_element(scores.data(), scores.data()+numInfer, scores.data()+scores.size(), greater<val>());
            threshold = scores(numInfer);
            // use tuples to track entity indices in H
            unsigned k = 0;
            for (unsigned j = 0; j < R.rows(); j++) {
                if ( f(j) > threshold )
                    top_pairs[k++] = tuple<int, val> (j, f(j));
                if ( k == numInfer ) break;
            }
            sort(top_pairs.begin(), top_pairs.begin()+k, tuple_comparator); 
            gid = g.id_of.at(i);
            for (unsigned j = 0; j < k; j++)
                ofs << gid << " " << h.id_of.at(get<0>(top_pairs[j])) << " " << get<1>(top_pairs[j]) << endl;
        }
        ofs.close();
    }
}

const char *get_full_path(const char* dir, const char* file_name) {
    char *p = new char[256];
    strcpy(p, dir);
    strcat(p, "/");
    strcat(p, file_name);
    return p;
}

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

    parser.add_option("-G", "--graphG") .metavar("FILE") .help("entity graph G on the left");
    parser.add_option("-H", "--graphH") .metavar("FILE") .help("entity graph H on the right");
    parser.add_option("-T", "--trainLinks") .metavar("FILE") .help("cross-graph links for training");
    parser.add_option("-V", "--validLinks") .metavar("FILE") .help("cross-graph links for validation");
    parser.add_option("-O", "--outDir") .metavar("FILE") .help("output directory\n");
    
    parser.add_option("-k", "--dimF") .type("int") .set_default(50) .help("inner dimension of F (%default)");
    parser.add_option("-p", "--dimG") .type("int") .set_default(100) .help("inner dimension of G (%default)");
    parser.add_option("-q", "--dimH") .type("int") .set_default(100) .help("inner dimension of H (%default)");
    parser.add_option("-C", "--C") .type("double") .set_default(1.0) .help("C*loss_func(F) + regularization(F) (%default)\n");

    char const* const algs[] = { "top", "pmf", "grmf" };
    parser.add_option("--algorithm") .metavar("{top, pmf, grmf}") .choices(&algs[0], &algs[3]) .set_default("top") .help("algorithm (%default)");
    parser.add_option("--convergence") .type("double") .set_default(1e-3) .help("desired convergence rate (%default)");
    parser.add_option("--decay") .type("double") .set_default(1.0) .help("decay factor for infinite ramdom walk (%default)");
    parser.add_option("--alpha") .type("double") .set_default(0.5) .help("backtracking parameter: \\alpha (%default)");
    parser.add_option("--beta") .type("double") .set_default(0.5) .help("backtracking parameter: \\beta (%default)");
    parser.add_option("--PCGTolerance") .type("double") .set_default(1e-5) .help("PCG tolerance (%default)");
    parser.add_option("--PCGMaxIter") .type("int") .set_default(50) .help("max PCG iterations (%default)");
    parser.add_option("--eta0") .type("double") .set_default(1e-3) .help("PMF/GRMF learning rate (%default)");
    parser.add_option("--maxThreads") .type("int") .set_default(4) .help("max number of training threads (%default)\n");

    parser.add_option("-i", "--doInfer") .action("store_true") .help("dump top [numInfer] induced cross-graph links associated with each entity in G");
    parser.add_option("--numInfer") .type("int") .set_default(10) .help("see above (%default)");

    Values& options = parser.parse_args(argc, argv);

    const char* graphG = (const char*) options.get("graphG");
    const char* graphH = (const char*) options.get("graphH");
    const char* trainLinks = (const char*) options.get("trainLinks");
    const char* validLinks = (const char*) options.get("validLinks");
    const char* outDir = (const char*) options.get("outDir");

    /*ensure all required file paths are specified*/
    if ( !(strlen(graphG) && strlen(graphH) && strlen(trainLinks) && strlen(validLinks) &&  strlen(outDir)) ) {
        parser.print_help();
        exit(1);
    }

    int stat = mkdir(outDir, 0777);
    if (stat != 0) {
        cerr << "Failed to create the output directory (does the dir already exist?)" << endl;
        exit(1);
    }

    const char* outPrediction = get_full_path(outDir, "prediction");
    const char* outInference = get_full_path(outDir, "inference");
    const char* outConfig = get_full_path(outDir, "config");

    Config opt;
    opt.k_f = (unsigned) options.get("dimF");
    opt.k_g = (unsigned) options.get("dimG");
    opt.k_h = (unsigned) options.get("dimH");
    opt.decay = (double) options.get("decay");
    opt.C = (double) options.get("C");
    opt.alg = (const char*) options.get("algorithm");
    opt.tol = (double) options.get("convergence");
    opt.alpha = (double) options.get("alpha");
    opt.beta = (double) options.get("beta");
    opt.pcgTol = (double) options.get("PCGTolerance");
    opt.pcgIter = (unsigned) options.get("PCGMaxIter");
    opt.eta0 = (double) options.get("eta0");

    /*load graph G on the left*/
    INFO_1("G <-- ", graphG) 
    Entity g(graphG);

    /*load graph H on the right*/
    INFO_1("H <-- ", graphH) 
    Entity h(graphH);

    /*load observed cross-graph links for training*/
    INFO_1("T <-- ", trainLinks) 
    Relation trn(trainLinks, g, h);

    /*load hold-out cross-graph links for testing*/
    INFO_1("V <-- ", validLinks)
    Relation val(validLinks, g, h);

    /*set the CPU threads for training*/
    unsigned maxThreads = (unsigned) options.get("maxThreads"); 
    unsigned defaultThreads = Eigen::nbThreads();
    unsigned ourThreads = maxThreads < defaultThreads ? maxThreads : defaultThreads;
    Eigen::setNbThreads(ourThreads);

    INFO_2("Entities in G: ", g.n)
    INFO_2("Entities in H: ", h.n)
    INFO_2("Training edges: ", trn.edges.size())
    INFO_2("Validation edges: ", val.edges.size())
    INFO_2("CPU threads: ", Eigen::nbThreads())

    /*initialize the algorithm*/
    Top top(opt);

    /*training*/
    assert(top.train(g, h, trn, val));

    /*dump the predictions to file*/
    INFO_1("Predictions --> ", outPrediction)
    Result result = top.predict(g, h, val, outPrediction);
    INFO_2("MAE", result.mae)
    INFO_2("RMSE", result.rmse)

    /*dump the inferred links as necessary*/
    if ( options.get("doInfer") ) {
        INFO_1("Induced Links --> ", outInference)
        dump_inference(outInference, (unsigned) options.get("numInfer"), &top, g, h);
    }

    ofstream ofs(outConfig);
    if ( !ofs.fail() ) {
        ofs 
            << "graphG" << " = " << graphG << endl
            << "graphH" << " = " << graphH << endl
            << "trainLinks" << " = " << trainLinks << endl
            << "validLinks" << " = " << validLinks << endl
            << "dimF" << " = " << opt.k_f << endl
            << "dimG" << " = " << opt.k_g << endl
            << "dimH" << " = " << opt.k_h << endl
            << "C" << " = " << opt.C << endl
            << "algorithm" << " = " << opt.alg << endl
            << "convergence" << " = " << opt.tol << endl
            << "decay" << " = " << opt.decay << endl
            << "alpha" << " = " << opt.alpha << endl
            << "beta" << " = " << opt.beta << endl
            << "PCGTolerance" << " = " << opt.pcgTol << endl
            << "PCGMaxIter" << " = " << opt.pcgIter << endl
            << "eta0" << " = " << opt.eta0 << endl
            << "maxThreads" << " = " << ourThreads << endl;
        ofs.close();
    }
    return 0;
}
