#ifndef _TOP_HH_
#define _TOP_HH_

#include "problem.hh"
#include "RedSVD.hh"
#include <ctime>
#include <string>

class Config {
public:
    unsigned k_f;
    unsigned k_g;
    unsigned k_h;
    val decay;
    val C;
    val tol;
    val alpha;
    val beta;
    val pcgTol;
    unsigned pcgIter;
    val eta0;
    std::string alg;
};

class Top
{
public:
    Top(const Config &opt);
    bool train(const Entity& e1, const Entity& e2, const Relation& trn, const Relation& tes);
    Result predict(const Entity& e1, const Entity& e2, const Relation& r, const char* output);
private:
    Config opt;
    mat inv_exp_lambda; /*inv exp of G's nonzero eigenvalues*/
    mat inv_exp_mu;     /*inv exp of H's nonzero eigenvalues*/
    mat U;
    mat V;
    mat L;
    mat R;
    sp_mat *loss_2nd;
    sp_mat *loss_1st;
    sp_mat *ratings;

    void initialize(const Entity& e1, const Entity& e2, const Relation& trn);
    Result validate(const Relation& r);
    /*
     * optimization objective:
     *      C*\|I.*(Y-F)\|_2^2 + 0.5*\vec(F)A\vec(F)
     * where F = LR', A is the adjacency matrix of the product graph
     */
    val objective(const mat& L, const mat& R, const Relation& r);
    /*
     *objective for probabilistic matrix factorization
     */
    val objective_PMF(const mat& L, const mat& R, const Relation& r);
    /*
     * 1st and 2nd order derivatives for the loss term
     */
    sp_mat *get_loss_1st(const mat& L, const mat& R, const Relation& r);
    sp_mat *get_loss_2nd(const mat& L, const mat& R, const Relation& r);
    /*
     * derivative of the objective function w.r.t. L and R
     */
    mat gradient_L(const mat& L, const mat& R, const Relation& r);
    mat gradient_R(const mat& L, const mat& R, const Relation& r);
    /*
     * the Hessian defines a mapping from F to some output matrix
     */
    mat hessian_map_L(const mat& L, const mat& R, const Relation& r, const mat& T);
    mat hessian_map_R(const mat& L, const mat& R, const Relation& r, const mat& T);
    /*
     * matrix-free conjugate gradient method for solving the linear system
     *      A\vec(X) = \vec(B)
     * where
     *      A = C*I + \sum \tau_{ij} (u_i \otimes v_j)(u_i \otimes v_j)^\top
     */
    mat matrix_pcg_L(const Relation& r, const mat& L0, const mat& R, const mat& B);
    mat matrix_pcg_R(const Relation& r, const mat& L, const mat& R0, const mat& B);
    /*
     * PCG preconditioning map
     */
    mat precondition_map_L(const mat&L, const mat& R, const Relation& r, const mat& inv_T);
    mat precondition_map_R(const mat&L, const mat& R, const Relation& r, const mat& inv_T);
    /*
     * return the symmetrically normalized adjacency matrix
     */
    sp_mat normalized_graph(const sp_mat& A);
};

#endif
