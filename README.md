Some examples for testing tensorflow-probability's BFGS algorithm.

Short overview:
- bfgs\_test\_script\_nonterminating.py: Initialize with simulated values; does not terminate
- bfgs\_test\_script\_failing.py: Initialize with (simulated values * 1.5); does terminate but fails to find MLE
- bfgs\_test\_script\_failing\_with\_hessian.py: Initialize with (simulated values * 1.5) and specify `initial_inverse_hessian_estimate`; does terminate but fails to find MLE


