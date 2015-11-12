#include "problem.hh"

bool Entity::saveSparseGraph(const char* filename) {
    std::ofstream ofs(filename);
    if (!ofs.fail()) {
        for (int k = 0; k < A.outerSize(); ++k) {
            for (sp_mat::InnerIterator it(A,k); it; ++it) {
                ofs << id_of.at(it.row()) << ' '
                    << id_of.at(it.col()) << ' '
                    << it.value() << std::endl;
            }
        }
        ofs.close();
        return true;
    }
    return false;
}

bool Entity::loadSparseGraph(const char* filename) {
    n = 0;
    std::string line;
    std::ifstream ifs(filename);
    std::string id_1, id_2;
    int i, j;
    if ( !ifs.fail() ) {
        /*index the entities*/
        int l = 0;
        while (getline(ifs, line)) {
            std::stringstream ss(line);
            ss >> id_1 >> id_2;
            if (!index_of.count(id_1)) index_of[id_1] = n++;
            if (!index_of.count(id_2)) index_of[id_2] = n++;
            l++;
        }
        /*inversely index the ids*/
        for (auto it = index_of.begin(); it != index_of.end(); ++it)
            id_of[it->second] = it->first;
        /*reset the file pointer*/
        ifs.clear();
        ifs.seekg(0);
        /*load the graph structure*/
        val v;
        std::vector<triplet> A_triplets(l*2);
        l = 0;
        while ( getline(ifs, line) ) {
            std::stringstream ss(line);
            ss >> id_1 >> id_2 >> v;
            i = index_of[id_1];
            j = index_of[id_2];
            /*make the graph symmetric*/
            A_triplets[l++] = triplet(i,j,v);
            A_triplets[l++] = triplet(j,i,v);
        }
        ifs.close();
        //add diagonal
        /*
         *for (int i = 0; i < n; i++) {
         *    A_triplets.push_back(triplet(i,i,1));
         *}
         */
        //construct sparse adjacency matrix from triplets
        A = sp_mat(n, n);
        A.setFromTriplets(A_triplets.begin(), A_triplets.end());
        for ( int i=0; i < A.outerSize(); ++i ) {
            for (sp_mat::InnerIterator it(A,i); it; ++it) {
                it.valueRef() = 1;
            }
        }
        return true;
    }
    return false;
}

void Entity::formCosKNNGraph(const sp_mat& X, int k) {
    sp_mat tmp = X*X.transpose();
    mat K = tmp;
    /*normalization*/
    mat d = K.diagonal().cwiseSqrt().cwiseInverse();
    mat Z = d.asDiagonal()*K*d.asDiagonal();
    mat g;
    std::vector<triplet> A_triplets;
    for ( int i = 0; i < n; i++ ) {
        g = Z.row(i);
        nth_element(g.data(),g.data()+k+1,g.data()+g.size(),std::greater<val>());
        val thres = g(k+1);
        for ( int j = 0; j < n; j++ ) {
            if ( Z(i,j) > thres && i != j) {
                /*make the graph symmetric*/
                A_triplets.push_back(triplet(i,j,1));
                A_triplets.push_back(triplet(j,i,1));
            }
        }
    }
    A = sp_mat(n,n);
    A.setFromTriplets(A_triplets.begin(), A_triplets.end());
    /*
     * construct the unweighted knn graph,
     * i.e. iterate over nonzero elements of A and set them to be 1
     */
    for ( int i=0; i < A.outerSize(); ++i ) {
        for (sp_mat::InnerIterator it(A,i); it; ++it) {
            it.valueRef() = 1;
        }
    }
}

sp_mat Entity::load(const char* filename) {
    n = 0;
    int p = 0;
    std::ifstream ifs(filename);
    sp_mat X;
    if ( !ifs.fail() ) {
        char delimiter;      // ":"
        int j;               // feature index
        val v;               // feature value
        std::string id;      // instance id
        std::vector<triplet> X_triplets;
        std::string line;
        while ( getline(ifs, line) ) {
            std::stringstream ss(line);
            ss >> id;
            id_of[n] = id;
            index_of[id] = n;
            while ( ss >> j >> delimiter >> v ) {
                if ( j > p ) p = j;
                X_triplets.push_back(triplet(n,j-1,v));
            } n++;
        }
        ifs.close();
        X = sp_mat(n, p);
        X.setFromTriplets(X_triplets.begin(), X_triplets.end());
    }
    return X;
}

bool Relation::load(const char* filename, const Entity& e1, const Entity& e2) {
    std::ifstream ifs(filename);
    if ( !ifs.fail() ) {
        val weight;
        std::string e1_id, e2_id;
        std::string line;
        while ( getline(ifs, line) ) {
            std::stringstream ss(line);
            ss >> e1_id;
            ss >> e2_id;
            ss >> weight;
            try {
                edges.push_back(triplet(e1.index_of.at(e1_id),e2.index_of.at(e2_id),weight));
            } catch (int e) {
                std::cerr << "out-of-sample entities: "
                    << e1_id << " " << e2_id << std::endl;
            }
        }
        ifs.close();
        return true;
    }
    return false;
}
