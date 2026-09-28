# Statistics original-versus-binding coverage

`oracle_statistics.cpp` compiles against pristine qm-dsp C sources; the Python
test feeds its recorded inputs to `pyqmdsp.statistics` and compares every output.

| API | Original execution recorded |
|---|---|
| PCA | Nonzero 36-by-3 data; 1, 2 and 3 retained components, including signs |
| Covariance inverse | Full 3-by-3 positive definite matrix, inverse and determinant |
| Gaussian/log Gaussian | Nonzero point and mean, correlated covariance |
| Forward/backward | Three states, four observations, complete posterior, transition posterior and log likelihood |
| HMM initialization | All initial/transition probabilities, means and covariance under fixed original time seed |
| HMM set parameters and decode | Prescribed two-state, two-dimensional model and all 40 decoded states |
| Baum–Welch update | Original posteriors fed to both; all updated parameters and decoded states |
| HMM train | Original stopping rule; all final parameters and decoded states after update then training |

HMM print is diagnostic output, not a numerical analysis. Close/context-manager
lifetime and rejected invalid inputs remain in the existing contract tests.
Only the randomized initialization comparison requires the Linux glibc fixed-time
test process. Prescribed-parameter inference/update/training is tested everywhere.
