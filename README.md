Some examples for testing tensorflow-probability's BFGS algorithm.

Please note that the simulated data is stored in `example.h5` and you will have to install `xarray` in order to load it.

Short overview:
- **bfgs\_test\_nonterminating.py:** <br>
  Initialize with simulated values; does not terminate
- **bfgs\_test\_failing.py:** <br>
  Initialize with (simulated values * 1.5); does terminate but fails to find MLE
- **bfgs\_test\_failing\_with\_hessian.py:** <br>
  Initialize with (simulated values * 1.5) and specify `initial_inverse_hessian_estimate`; does terminate but fails to find MLE


