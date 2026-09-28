// Original PCA and Gaussian HMM numerical outputs, independent of Python adapters.
// Reads: pristine qm-dsp PCA/HMM APIs and the fixture-only JSON writer.
#include "oracle.h"
#include "hmm/hmm.h"
#include "maths/pca/pca.h"
#include <algorithm>
#include <cstdlib>

using Matrix = std::vector<std::vector<double>>;
using Cube = std::vector<Matrix>;
struct Rows {
    Matrix values;
    std::vector<double *> pointers;
    explicit Rows(Matrix x): values(std::move(x)) {
        for(auto &row:values) pointers.push_back(row.data());
    }
    double **data() { return pointers.data(); }
};
struct Tensor {
    Cube values;
    std::vector<std::vector<double *>> rows;
    std::vector<double **> pointers;
    Tensor(int t,int n):values(t,Matrix(n,std::vector<double>(n))),rows(t) {
        for(int i=0;i<t;++i) {
            for(auto &row:values[i]) rows[i].push_back(row.data());
            pointers.push_back(rows[i].data());
        }
    }
};
void parameters(Json &out, const std::string &key, model_t *m) {
    Json p;
    p.set("initial",std::vector<double>(m->p0,m->p0+m->N));
    Matrix a(m->N),mu(m->N),cov(m->L);
    for(int i=0;i<m->N;++i) {
        a[i].assign(m->a[i],m->a[i]+m->N);
        mu[i].assign(m->mu[i],m->mu[i]+m->L);
    }
    for(int i=0;i<m->L;++i) cov[i].assign(m->cov[i],m->cov[i]+m->L);
    p.set("transition",a);p.set("means",mu);p.set("covariance",cov);
    out.set(key,p);
}
int main(int argc,char **argv) {
    if(argc!=2) return 2;
    Json input,out,random,root;
    Matrix data(36,std::vector<double>(3));
    for(int i=0;i<36;++i) {
        data[i][0]=3*std::sin(i*0.47)+0.2*i;
        data[i][1]=std::cos(i*0.63)+0.13*(i%5);
        data[i][2]=0.2*std::sin(i*1.2)+0.09*(i%7);
    }
    input.set("pca",data);
    for(int n=1;n<=3;++n) {
        Rows copy(data);
        pca_project(copy.data(),36,3,n);
        for(auto &r:copy.values) r.resize(n);
        out.set("pca_"+std::to_string(n),copy.values);
    }
    Matrix cov={{2.4,0.3,-0.1},{0.3,1.3,0.2},{-0.1,0.2,0.8}};
    std::vector<double> point={0.31,-0.77,1.28},mean={0.1,-0.2,0.3},y(3),z(3);
    input.set("covariance",cov);input.set("point",point);input.set("mean",mean);
    Rows c(cov),inv(Matrix(3,std::vector<double>(3)));
    double determinant;
    invert(c.data(),3,inv.data(),&determinant);
    out.set("inverse",inv.values);out.set("determinant",determinant);
    out.set("gaussian",gauss(point.data(),3,mean.data(),inv.data(),determinant,y.data(),z.data()));
    out.set("log_gaussian",loggauss(point.data(),3,mean.data(),inv.data(),determinant,y.data(),z.data()));

    std::vector<double> initial={0.6,0.25,0.15};
    Matrix transition={{0.7,0.2,0.1},{0.2,0.6,0.2},{0.1,0.2,0.7}};
    Matrix emissions={{0.2,0.5,0.8},{0.3,0.7,0.1},{0.8,0.2,0.15},{0.1,0.2,0.8}};
    input.set("initial",initial);input.set("transition",transition);input.set("emissions",emissions);
    Rows a(transition),b(emissions),gamma(Matrix(4,std::vector<double>(3)));
    Tensor xi(4,3);
    double ll=0,ll1=0,ll2=0;
    forward_backwards(xi.pointers.data(),gamma.data(),&ll,&ll1,&ll2,1,3,4,initial.data(),a.data(),b.data());
    xi.values.resize(3);
    out.set("posterior",gamma.values);out.set("transition_posterior",xi.values);out.set("log_likelihood",ll);

    Matrix observations(40,std::vector<double>(2));
    for(int i=0;i<40;++i) {
        bool second=(i/10)%2;
        observations[i][0]=(second?1.8:-1.5)+0.27*std::sin(i*1.3);
        observations[i][1]=(second?0.8:-0.7)+0.21*std::cos(i*0.79);
    }
    input.set("observations",observations);
    Rows x(observations);
    model_t *model=hmm_init(x.data(),40,2,2);
    parameters(random,"initial_parameters",model);
    std::vector<double> hp={0.55,0.45};
    Matrix ha={{0.85,0.15},{0.12,0.88}},hm={{-1.5,-0.7},{1.8,0.8}},hc={{0.7,0.1},{0.1,0.6}};
    Json prescribed;
    prescribed.set("initial",hp);prescribed.set("transition",ha);prescribed.set("means",hm);prescribed.set("covariance",hc);
    input.set("prescribed",prescribed);
    std::copy(hp.begin(),hp.end(),model->p0);
    for(int i=0;i<2;++i) {
        std::copy(ha[i].begin(),ha[i].end(),model->a[i]);
        std::copy(hm[i].begin(),hm[i].end(),model->mu[i]);
        std::copy(hc[i].begin(),hc[i].end(),model->cov[i]);
    }
    parameters(out,"set_parameters",model);
    std::vector<int> labels(40);
    viterbi_decode(x.data(),40,model,labels.data());out.set("decode_before",labels);
    Rows emissions2(Matrix(40,std::vector<double>(2))),inv2(Matrix(2,std::vector<double>(2)));
    invert(model->cov,2,inv2.data(),&determinant);
    for(int t=0;t<40;++t) for(int state=0;state<2;++state)
        emissions2.values[t][state]=gauss(x.values[t].data(),2,model->mu[state],inv2.data(),determinant,y.data(),z.data());
    Rows gamma2(Matrix(40,std::vector<double>(2)));
    Tensor xi2(40,2);
    ll=ll1=ll2=0;
    forward_backwards(xi2.pointers.data(),gamma2.data(),&ll,&ll1,&ll2,1,2,40,model->p0,model->a,emissions2.data());
    input.set("update_posterior",gamma2.values);
    Cube xi_input=xi2.values;xi_input.resize(39);input.set("update_transition_posterior",xi_input);
    baum_welch(model->p0,model->a,model->mu,model->cov,2,40,2,x.data(),xi2.pointers.data(),gamma2.data());
    parameters(out,"after_update",model);
    viterbi_decode(x.data(),40,model,labels.data());out.set("decode_updated",labels);
    hmm_train(x.data(),40,model);parameters(out,"after_train",model);
    viterbi_decode(x.data(),40,model,labels.data());out.set("decode_trained",labels);
    hmm_close(model);
    root.set("inputs",input);root.set("outputs",out);root.set("stochastic",random);root.save(argv[1]);
}
