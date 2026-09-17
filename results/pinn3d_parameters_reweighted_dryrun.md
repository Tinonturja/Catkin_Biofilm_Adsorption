# PINN3D Learned Parameters (loss-reweighting dry run, cloud sandbox)

This is a verification run of the fixed-loss-weight fix, executed in a cloud
sandbox (not your machine) to confirm the code runs and to check whether the
loss-spike pattern improves. Re-run train.py on your own machine for your
canonical results/pinn3d_training_results.csv and results/pinn3d_parameters.md.

- Epochs: 8000

## Physical Parameters

- Rate Constant (K2): 0.004678
- Freundlich Constant (Kf): 0.406240
- Freundlich Exponent (N): 0.417347

## Final Losses (raw, unweighted -- comparable to the pre-fix run)

- Data Loss: 0.000000
- PSO Loss: 0.000075
- Initial Loss: 0.000003
- ISO Loss: 0.000027
- Total Loss (unweighted): 0.000104
- Weighted Total Loss (optimizer objective): 0.000404
