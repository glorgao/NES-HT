
### Official Implementation for NESHT

This project implements an evolution algorithm that integrates the hard-thresholding (HT) operator into the well-known natural evolution strategies (NES) algorithm. For the paper, please check [this arXiv link].

### Details

- Parallelism. We employ the simple <code>joblib</code> package, finding it much faster than using <code>ray</code> on a single node cluster. (Our single node has 512 cores.)
- Policy Network. We apply the NESHT agent to a single linear layer agent, as it's well established that the capacity of a single linear layer suffices for Mujoco tasks.
- Gaussian Noise. Our comparisons are conducted on noisy Mujoco environments, where the state is a combination of Gaussian noise and environment-provided observations.
- For any questions about the implementation, please don't hesitate to contact me.
