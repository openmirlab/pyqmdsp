# Statistics coverage

| Upstream public API | Python API | Evidence |
|---|---|---|
| maths/pca/pca.h: pca_project | statistics.pca | Centering, orthogonal components and conserved full-dimensional variance |
| hmm/hmm.h: hmm_init, hmm_close, model_t | statistics.HMM constructor/close, parameters/set_parameters | Shape, probability and covariance validation; idempotent close |
| hmm_train | HMM.train | Real Gaussian training and normalized transitions |
| viterbi_decode | HMM.decode | Known two-state sequence |
| forward_backwards | statistics.forward_backward | Exact enumeration of all state paths |
| baum_welch | HMM.update | Explicit posterior update followed by training |
| invert | statistics.covariance_inverse | NumPy inverse/determinant reference |
| gauss, loggauss | statistics.gaussian(logarithmic=...) | Independent Gaussian closed form |
| hmm_print | HMM.print; HMM.parameters | Native stdout or structured copies |

Input covariance must be symmetric positive definite: the upstream routine is
intended for covariance and does not correctly account for arbitrary determinant
signs. HMM initialization preserves the upstream time-seeded RNG. HMM calls keep
the GIL, serializing global RNG access. Training follows upstream stopping behavior;
collapsed covariance raises a Python exception. The upstream allocation omissions
in hmm_close and invert are fixed by a documented build-time patch, without
changing calculations. PCA returns the first requested projected columns.
