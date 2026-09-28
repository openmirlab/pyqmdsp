// HMM and PCA ownership adapters; calculations stay in the pinned upstream.
// Reads: hmm/hmm.h, maths/pca/pca.h, common.h. GIL serializes upstream global RNG.
#include "common.h"
#include "hmm/hmm.h"
#include "maths/pca/pca.h"
#include <algorithm>
#include <cstdlib>
#include <numeric>
using Matrix = std::vector<std::vector<double>>;
using Cube = std::vector<Matrix>;
namespace {
struct Rows {
    Matrix values;
    std::vector<double *> ptrs;
    explicit Rows(Matrix x) : values(std::move(x)) {
        for (auto &row : values) ptrs.push_back(row.data());
    }
    double **data() { return ptrs.data(); }
};
void shape(const Matrix &x, size_t rows, size_t cols) {
    if (x.size() != rows || rows == 0 || cols == 0) throw std::invalid_argument("invalid matrix shape");
    for (auto &r : x) {
        if (r.size() != cols) throw std::invalid_argument("matrix rows must have the expected size");
        for (double v : r) if (!std::isfinite(v)) throw std::invalid_argument("matrix must be finite");
    }
}
void positive_covariance(const Matrix &x) {
    shape(x, x.size(), x.size());
    Matrix l(x.size(), std::vector<double>(x.size()));
    for (size_t i=0;i<x.size();++i) for(size_t j=0;j<=i;++j) {
        if (std::abs(x[i][j]-x[j][i])>1e-10*(1+std::abs(x[i][j]))) throw std::invalid_argument("covariance must be symmetric");
        double v=x[i][j];
        for(size_t k=0;k<j;++k) v-=l[i][k]*l[j][k];
        if(i==j) {
            if (!(v>0)) throw std::invalid_argument("covariance must be positive definite");
            l[i][j]=std::sqrt(v);
        } else l[i][j]=v/l[j][j];
    }
}
void probability(const std::vector<double> &x) {
    double sum=0;
    for(double v:x) { if(!std::isfinite(v)||v<0) throw std::invalid_argument("probabilities must be finite and nonnegative"); sum+=v; }
    if(std::abs(sum-1)>1e-8) throw std::invalid_argument("probabilities must sum to one");
}
struct Tensor {
    Cube values;
    std::vector<std::vector<double *>> rows;
    std::vector<double **> planes;
    explicit Tensor(Cube x): values(std::move(x)) {
        for(auto &mat:values) { rows.emplace_back(); for(auto &row:mat) rows.back().push_back(row.data()); }
        for(auto &row:rows) planes.push_back(row.data());
    }
    double ***data() {return planes.data();}
};
class HMM {
    model_t *model=nullptr;
    void ready() const { if(!model) throw std::runtime_error("HMM is closed"); }
    Matrix covariance() const {
        Matrix out(model->L,std::vector<double>(model->L));
        for(int i=0;i<model->L;++i) std::copy_n(model->cov[i],model->L,out[i].data());
        return out;
    }
public:
    HMM(Matrix x,int states) {
        if(x.size()<2 || x.empty() || x[0].empty() || states<1 || static_cast<size_t>(states)>x.size()) throw std::invalid_argument("need at least two observations and 1 <= states <= observations");
        shape(x,x.size(),x[0].size());
        for(size_t j=0;j<x[0].size();++j) {
            bool differs=false; for(auto &r:x) if(r[j]!=x[0][j]) differs=true;
            if(!differs) throw std::invalid_argument("each feature must have nonzero variance");
        }
        Rows rows(std::move(x));
        model=hmm_init(rows.data(),int(rows.values.size()),int(rows.values[0].size()),states);
    }
    ~HMM(){close();}
    void close(){if(model){hmm_close(model);model=nullptr;}}
    nb::dict parameters() const {
        ready(); nb::dict d;
        std::vector<double> initial(model->p0,model->p0+model->N);
        Matrix transitions(model->N,std::vector<double>(model->N)), means(model->N,std::vector<double>(model->L));
        for(int i=0;i<model->N;++i){std::copy_n(model->a[i],model->N,transitions[i].data());std::copy_n(model->mu[i],model->L,means[i].data());}
        d["initial"]=nb::cast(initial);d["transition"]=nb::cast(transitions);d["means"]=nb::cast(means);d["covariance"]=nb::cast(covariance());return d;
    }
    void set_parameters(std::vector<double> initial,Matrix transitions,Matrix means,Matrix cov){
        ready(); if(initial.size()!=size_t(model->N))throw std::invalid_argument("initial shape mismatch");
        probability(initial);shape(transitions,model->N,model->N);for(auto&r:transitions)probability(r);
        shape(means,model->N,model->L);shape(cov,model->L,model->L);positive_covariance(cov);
        std::copy(initial.begin(),initial.end(),model->p0);
        for(int i=0;i<model->N;++i){std::copy(transitions[i].begin(),transitions[i].end(),model->a[i]);std::copy(means[i].begin(),means[i].end(),model->mu[i]);}
        for(int i=0;i<model->L;++i)std::copy(cov[i].begin(),cov[i].end(),model->cov[i]);
    }
    void train(Matrix x){
        ready();shape(x,x.size(),model->L);if(x.size()<2)throw std::invalid_argument("training needs at least two observations");positive_covariance(covariance());
        Rows rows(std::move(x));hmm_train(rows.data(),int(rows.values.size()),model);positive_covariance(covariance());
    }
    std::vector<int> decode(Matrix x){
        ready();shape(x,x.size(),model->L);positive_covariance(covariance());Rows rows(std::move(x));std::vector<int> q(rows.values.size());viterbi_decode(rows.data(),int(q.size()),model,q.data());return q;
    }
    void update(Matrix x,Matrix gamma,Cube xi){
        ready();shape(x,x.size(),model->L);const size_t t=x.size(),n=model->N;
        if(t<2||xi.size()!=t-1)throw std::invalid_argument("xi must have observations - 1 planes");
        shape(gamma,t,n);for(auto&r:gamma)probability(r);
        for(size_t j=0;j<n;++j){double s=0;for(auto&r:gamma)s+=r[j];if(s<=0)throw std::invalid_argument("each state must have positive posterior mass");}
        for(auto &plane:xi){shape(plane,n,n);std::vector<double> flat;for(auto&r:plane)flat.insert(flat.end(),r.begin(),r.end());probability(flat);}
        Rows obs(std::move(x)),post(std::move(gamma));Tensor pairs(std::move(xi));
        baum_welch(model->p0,model->a,model->mu,model->cov,model->N,int(t),model->L,obs.data(),pairs.data(),post.data());positive_covariance(covariance());
    }
    void print(){ready();hmm_print(model);}
};
}
void bind_statistics(nb::module_ &m){
    m.def("statistics_pca",[](Matrix x,int components){
        if(x.size()<2||x[0].empty()||components<1||size_t(components)>x[0].size())throw std::invalid_argument("PCA requires >=2 rows and 1 <= components <= columns");
        shape(x,x.size(),x[0].size());Rows rows(std::move(x));
        {nb::gil_scoped_release release;pca_project(rows.data(),int(rows.values.size()),int(rows.values[0].size()),components);}
        for(auto &r:rows.values)r.resize(components);return rows.values;
    });
    m.def("statistics_invert",[](Matrix x){positive_covariance(x);Rows in(std::move(x)),out(Matrix(in.values.size(),std::vector<double>(in.values.size())));double det;invert(in.data(),int(in.values.size()),out.data(),&det);return std::make_pair(out.values,det);});
    m.def("statistics_gaussian",[](std::vector<double>x,std::vector<double>mean,Matrix cov,bool logarithmic){
        shape({x},1,x.size());shape({mean},1,x.size());shape(cov,x.size(),x.size());positive_covariance(cov);
        Rows c(std::move(cov)),inv(Matrix(x.size(),std::vector<double>(x.size())));double det;invert(c.data(),int(x.size()),inv.data(),&det);std::vector<double>y(x.size()),z(x.size());
        return logarithmic?loggauss(x.data(),int(x.size()),mean.data(),inv.data(),det,y.data(),z.data()):gauss(x.data(),int(x.size()),mean.data(),inv.data(),det,y.data(),z.data());
    });
    m.def("statistics_forward_backward",[](std::vector<double>p,Matrix a,Matrix b){
        if(p.empty())throw std::invalid_argument("initial cannot be empty");probability(p);shape(a,p.size(),p.size());for(auto&r:a)probability(r);shape(b,b.size(),p.size());
        for(auto&r:b)for(double v:r)if(v<=0)throw std::invalid_argument("emission probabilities must be positive");
        size_t t=b.size(),n=p.size();Rows transitions(std::move(a)),emissions(std::move(b)),gamma(Matrix(t,std::vector<double>(n)));Tensor xi(Cube(t,Matrix(n,std::vector<double>(n))));
        double ll=0,l1=0,l2=0;forward_backwards(xi.data(),gamma.data(),&ll,&l1,&l2,1,int(n),int(t),p.data(),transitions.data(),emissions.data());xi.values.resize(t-1);
        return std::make_tuple(gamma.values,xi.values,ll);
    });
    nb::class_<HMM>(m,"NativeHMM")
        .def(nb::init<Matrix,int>()).def("parameters",&HMM::parameters)
        .def("set_parameters",&HMM::set_parameters).def("train",&HMM::train)
        .def("decode",&HMM::decode).def("update",&HMM::update)
        .def("print",&HMM::print).def("close",&HMM::close);
}
