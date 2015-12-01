#include "Top.hh"

Top::Top(const Config& _opt) { this->opt = _opt; }

sp_mat Top::normalized_graph(const sp_mat& A) {
    mat degree = A*mat::Ones(A.rows(),1);
    /*in case of zero-degree vertices*/
    for (unsigned i = 0; i < degree.rows(); i++)
        if (degree(i,0) == 0)
            degree.coeffRef(i,0) = 1;
    mat inv_degree = degree.cwiseSqrt().cwiseInverse();
    return inv_degree.asDiagonal()*A*inv_degree.asDiagonal();
}

val Top::objective(const mat& L, const mat& R, const Relation& r) {
    val sq_err = 0, s;
    for (auto it = r.edges.cbegin(); it != r.edges.cend(); ++it) {
        s = L.row(it->row()).dot(R.row(it->col())) - it->value();
        sq_err += s*s;
    }
    mat LtU = L.transpose()*U;
    mat RtV = R.transpose()*V;
    mat RtR = R.transpose()*R;
    mat LtL = L.transpose()*L;
    mat U_L = LtU * inv_exp_lambda.asDiagonal() * LtU.transpose();
    mat V_R = RtV * inv_exp_mu.asDiagonal() * RtV.transpose();
    return opt.C * sq_err + 0.5 * ((U_L + LtL)*(V_R + RtR)).trace();
}

val Top::objective_PMF(const mat& L, const mat& R, const Relation& r) {
    val sq_err = 0, s;
    for (auto it = r.edges.cbegin(); it != r.edges.cend(); ++it) {
        s = L.row(it->row()).dot(R.row(it->col())) - it->value();
        sq_err += s*s;
    }
    return opt.C * sq_err + 0.5 * (L.squaredNorm() + R.squaredNorm());
}

val Top::objective_GRMF(const mat& L, const mat& R, const Relation& r) {
    val sq_err = 0, s;
    for (auto it = r.edges.cbegin(); it != r.edges.cend(); ++it) {
        s = L.row(it->row()).dot(R.row(it->col())) - it->value();
        sq_err += s*s;
    }
    mat LtU = L.transpose() * U;
    mat RtV = R.transpose() * V;
    return opt.C * sq_err + 0.5 * (
            L.squaredNorm() - (LtU * this->lambda.asDiagonal() * LtU.transpose()).trace() +
            R.squaredNorm() - (RtV * this->mu.asDiagonal() * RtV.transpose()).trace());
}

sp_mat* Top::get_loss_1st(const mat& L, const mat& R, const Relation& r) {
#pragma omp parallel for
    for (unsigned k = 0; k < loss_1st->outerSize(); ++k) {
        for (sp_mat::InnerIterator it(*loss_1st, k); it; ++it)
            it.valueRef() = L.row(it.row()).dot(R.row(it.col()));
    }
    (*loss_1st) -= (*ratings);
    return loss_1st;
}

sp_mat* Top::get_loss_2nd(const mat& L, const mat& R, const Relation& r) {
#pragma omp parallel for
    for (unsigned k = 0; k < loss_2nd->outerSize(); ++k) {
        for (sp_mat::InnerIterator it(*loss_2nd, k); it; ++it)
            it.valueRef() = L.row(it.row()).dot(R.row(it.col()));
    }
    return loss_2nd;
}

mat Top::gradient_L(const mat& L, const mat& R, const Relation& r) {
    mat VtR = V.transpose()*R;
    mat RtR = R.transpose()*R;
    mat UtL = U.transpose()*L;
    mat T = VtR.transpose() * inv_exp_mu.asDiagonal() * VtR + RtR; 
    return 2 * opt.C * (*get_loss_1st(L, R, r)) * R + (U * (inv_exp_lambda.asDiagonal() * UtL) + L) * T;
}

mat Top::gradient_R(const mat& L, const mat& R, const Relation& r) {
    mat UtL = U.transpose()*L;
    mat LtL = L.transpose()*L;
    mat VtR = V.transpose()*R;
    mat T = UtL.transpose() * inv_exp_lambda.asDiagonal() * UtL + LtL;
    return 2 * opt.C * (*get_loss_1st(L, R, r)).transpose() * L + (V * (inv_exp_mu.asDiagonal() * VtR) + R) * T;
}

mat Top::hessian_map_L(const mat& L, const mat& R, const Relation& r, const mat& T) {
    mat UtL = U.transpose()*L;
    return 2 * opt.C * (*get_loss_2nd(L, R, r)) * R + (U * (inv_exp_lambda.asDiagonal() * UtL) + L) * T;
}

mat Top::precondition_map_L(const mat&L, const mat& R, const Relation& r, const mat& inv_T) {
    mat UtL = U.transpose()*L;
    mat exp_lambda_plus_1 = inv_exp_lambda.cwiseInverse().array() + 1;
    return (L - U * (exp_lambda_plus_1.asDiagonal() * UtL)) * inv_T;
}

mat Top::hessian_map_R(const mat& L, const mat& R, const Relation& r, const mat& T) {
    mat VtR = V.transpose()*R;
    return 2 * opt.C * (*get_loss_2nd(L, R, r)).transpose() * L + (V * (inv_exp_mu.asDiagonal() * VtR) + R) * T;
}

mat Top::precondition_map_R(const mat& L, const mat& R, const Relation& r, const mat& inv_T) {
    mat VtR = V.transpose()*R;
    mat exp_mu_plus_1 = inv_exp_mu.cwiseInverse().array() + 1;
    return (R - V * (exp_mu_plus_1.asDiagonal() * VtR)) * inv_T;
}

val inner_dot(const mat& A, const mat& B) {
    /*
     *val s = 0;
     *#pragma omp parallel for reduction(+:s)
     *for(int i = 0; i < A.rows(); i++) {
     *    s += A.row(i).dot(B.row(i));
     *}
     *return s;
     */
    return A.cwiseProduct(B).sum();
}

mat Top::matrix_pcg_L(const Relation& r, const mat& L0, const mat& R, const mat& B) {
    /*precompute T for the hessian/precondition map*/
    mat VtR = V.transpose()*R;
    mat RtR = R.transpose()*R;
    mat T = VtR.transpose() * inv_exp_mu.asDiagonal() * VtR + RtR;
    mat inv_T = T.inverse();
    /*initialize PCG*/
    mat L = L0;
    mat E = B - hessian_map_L(L, R, r, T);
    val res0 = E.norm();
    mat Z = precondition_map_L(E, R, r, inv_T);
    mat P = Z;
    mat AP;
    val alpha_num, alpha_den, alpha, beta, conv;
    for (unsigned i = 0; i < opt.pcgIter; i++) {
        alpha_num = inner_dot(E, Z);
        AP = hessian_map_L(P, R, r, T);
        alpha_den = inner_dot(P, AP);
        alpha = alpha_num/alpha_den;
        L += alpha*P;
        E -= alpha*AP;
        conv = E.norm()/res0;
        fprintf(stderr, "L.PCG it %2d residual %e.\r", i+1, conv);
        if (conv < opt.pcgTol)
            break;
        Z = precondition_map_L(E, R, r, inv_T);
        beta = inner_dot(E, Z)/alpha_num;
        P = Z + beta*P;
    }
    return L;
}

mat Top::matrix_pcg_R(const Relation& r, const mat& L, const mat& R0, const mat& B) {
    /*precompute T for hessian/precondition map*/
    mat UtL = U.transpose()*L;
    mat LtL = L.transpose()*L;
    mat T =  UtL.transpose() * inv_exp_lambda.asDiagonal() * UtL + LtL;
    mat inv_T = T.inverse();
    /*initialize PCG*/
    mat R = R0;
    mat E = B - hessian_map_R(L, R, r, T);
    val res0 = E.norm();
    mat Z = precondition_map_R(L, E, r, inv_T);
    mat P = Z;
    mat AP;
    val alpha_num, alpha_den, alpha, beta, conv;
    for (unsigned i = 0; i < opt.pcgIter; i++) {
        alpha_num = inner_dot(E, Z);
        AP = hessian_map_R(L, P, r, T);
        alpha_den = inner_dot(P, AP);
        alpha = alpha_num/alpha_den;
        R += alpha*P;
        E -= alpha*AP;
        conv = E.norm()/res0;
        fprintf(stderr, "R.PCG it=%2d residual=%e.\r", i+1, conv);
        if (conv < opt.pcgTol)
            break;
        Z = precondition_map_R(L, E, r, inv_T);
        beta = inner_dot(E, Z)/alpha_num;
        P = Z + beta*P;
    }
    return R;
}

Result Top::validate(const Relation& r) {
    val se = 0;
    val ae = 0;
    val delta, score;
    for (auto it = r.edges.cbegin(); it != r.edges.cend(); ++it) {
        score = L.row(it->row()).dot(R.row(it->col()));
        delta = it->value() - score;
        se += delta*delta;
        ae += fabs(delta);
    }
    Result result;
    result.rmse = sqrt(se / r.edges.size());
    result.mae = ae / r.edges.size();
    return result;
}

Result Top::predict(const Entity& e1, const Entity& e2, const Relation& r, const char* output) {
    std::ofstream ofs(output);
    assert(!ofs.fail());
    unsigned i, j;
    val se = 0;
    val ae = 0;
    val delta, score;
    /*dump the predictions on test set for evaluation*/
    for (auto it = r.edges.cbegin(); it != r.edges.cend(); ++it) {
        i = it->row();
        j = it->col();
        score = L.row(i).dot(R.row(j));
        delta = it->value() - score;
        ofs << e1.id_of.at(i) << ' '
            << e2.id_of.at(j) << ' '
            << score << std::endl;
        se += delta*delta;
        ae += fabs(delta);
    }
    ofs.close();
    Result result;
    result.mae = ae / r.edges.size();
    result.rmse = sqrt(se / r.edges.size());
    return result;
}

void Top::initialize(const Entity& e1, const Entity& e2, const Relation& trn) {
    /*
     * Though normalized, the largeset eigenvalue may not be exactly 1,
     * as RedSVD is an approximate SVD solver based on column sampling 
     */
    if (opt.alg == "top" || opt.alg == "grmf") {
        fprintf(stderr, "Approximating the eigensystem ... ");
        RedSVD::RedSVD<sp_mat> svd_g(normalized_graph(e1.A), this->opt.k_g);
        RedSVD::RedSVD<sp_mat> svd_h(normalized_graph(e2.A), this->opt.k_h);
        fprintf(stderr, "Done.\n");

        this->U = svd_g.matrixU(); 
        this->V = svd_h.matrixU();

        this->lambda = svd_g.singularValues();
        this->mu = svd_h.singularValues();
        this->inv_exp_lambda = (this->opt.decay * this->lambda).array().exp().cwiseInverse();
        this->inv_exp_mu = (this->opt.decay * this->mu).array().exp().cwiseInverse();
        this->inv_exp_lambda.array() -= 1; // XXX: Trick
        this->inv_exp_mu.array() -= 1;
    }

    unsigned m = e1.n;
    unsigned n = e2.n;

    /*initialize L*R' = F*/
    this->L = mat::Random(m, this->opt.k_f);
    this->R = mat::Random(n, this->opt.k_f);

    /*initialize auxilliary vairables for faster optimization subroutine*/
    std::vector<triplet> triplets(trn.edges.size());
    unsigned k = 0;
    for (auto it = trn.edges.cbegin(); it != trn.edges.cend(); ++it)
        triplets[k++] = triplet(it->row(),it->col(),it->value());
    /*to store \ell*/
    this->ratings = new sp_mat(m, n);
    this->ratings->setFromTriplets(triplets.begin(), triplets.end());
    /*to store \nabla \ell*/
    this->loss_1st = new sp_mat(m, n);
    this->loss_1st->setFromTriplets(triplets.begin(), triplets.end());
    /*to store \nabla^2 \ell*/
    this->loss_2nd = new sp_mat(m, n);
    this->loss_2nd->setFromTriplets(triplets.begin(), triplets.end());
}

bool Top::train(const Entity& e1, const Entity& e2, const Relation& trn, const Relation& tes) {
    initialize(e1, e2, trn);
    val obj_new = 0, obj_old = 0;

    if (opt.alg == "top") obj_old = objective(L, R, trn);
    if (opt.alg == "pmf") obj_old = objective_PMF(L, R, trn);
    if (opt.alg == "grmf") obj_old = objective_GRMF(L, R, trn);

    mat nabla_L, nabla_R, delta_L, delta_R;
    val t, conv;
    unsigned iter = 0;
    std::clock_t start;
    printf("%-5s %-12s %-10s %-6s %-15s %-15s\n", "ITER", "OBJ", "CONV", "SECS", "TRN.[RMSE,MAE]", "VAL.[RMSE,MAE]");

    do {
        start = std::clock();
        if (opt.alg == "top") { /*Top*/
            /*Update L*/ {
                /*compute the gradient*/
                nabla_L = gradient_L(L, R, trn);
                /*compute newton direction*/
                delta_L = matrix_pcg_L(trn, L, R, nabla_L);
                /*backtracking for the dumped Newton step*/
                for (t = 1; objective(L - t*delta_L, R, trn) >
                        obj_old - opt.alpha*t*inner_dot(nabla_L, delta_L); t *= opt.beta); 
                L -= t*delta_L;
            }
            /*Update R*/ {
                /*compute the gradient*/
                nabla_R = gradient_R(L, R, trn);
                /*compute newton direction*/
                delta_R = matrix_pcg_R(trn, L, R, nabla_R);
                /*backtracking for the dumped Newton step*/
                for (t = 1; objective(L, R - t*delta_R, trn) >
                        obj_old - opt.alpha*t*inner_dot(nabla_R, delta_R); t *= opt.beta); 
                R -= t*delta_R;
            }
            obj_new = objective(L, R, trn);
        }
        if (opt.alg == "pmf") { /*Probabilistic Matrix Factorization*/
            /*Update L*/ {
                nabla_L = 2 * opt.C * (*get_loss_1st(L, R, trn)) * R + L;
                for (t = opt.eta0; objective_PMF(L - t*nabla_L, R, trn) >
                        obj_old - opt.alpha*t*nabla_L.squaredNorm(); t *= opt.beta);
                L -= t*nabla_L;
            }
            /*Update R*/ {
                nabla_R = 2 * opt.C * (*get_loss_1st(L, R, trn)).transpose() * L + R;
                for (t = opt.eta0; objective_PMF(L, R - t*nabla_R, trn) >
                        obj_old - opt.alpha*t*nabla_R.squaredNorm(); t *= opt.beta);
                R -= t*nabla_R;
            }
            obj_new = objective_PMF(L, R, trn);
        }
        if (opt.alg == "grmf") { /*Graph Regularized Matrix Factorization*/
            /*Update L*/ {
                nabla_L = 2 * opt.C * (*get_loss_1st(L, R, trn)) * R
                    + L - U * this->lambda.asDiagonal() * (U.transpose() * L);
                for (t = opt.eta0; objective_GRMF(L - t*nabla_L, R, trn) >
                        obj_old - opt.alpha*t*nabla_L.squaredNorm(); t *= opt.beta);
                L -= t*nabla_L;
            }
            /*Update R*/ {
                nabla_R = 2 * opt.C * (*get_loss_1st(L, R, trn)).transpose() * L
                    + R - V * this->mu.asDiagonal() * (V.transpose() * R);
                for (t = opt.eta0; objective_GRMF(L, R - t*nabla_R, trn) >
                        obj_old - opt.alpha*t*nabla_R.squaredNorm(); t *= opt.beta);
                R -= t*nabla_R;
            }
            obj_new = objective_GRMF(L, R, trn);
        }
        /*info disp and workflow control*/
        conv = (obj_old - obj_new) / obj_old;
        Result trn_res = validate(trn);
        Result tes_res = validate(tes);
        printf("%-5d %-12.5e %-10.3e %-6.3f %-7.4f %-7.4f %-7.4f %-7.4f\n",
                ++iter, obj_new, conv, (std::clock() - start) / (double) CLOCKS_PER_SEC, trn_res.rmse, trn_res.mae, tes_res.rmse, tes_res.mae);
        obj_old = obj_new;
    } while (conv > opt.tol);
    return true;
}
