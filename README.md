
### Official Implementation for NESHT(Hard-Thresholding Meets Evolution Strategies in Reinforcement Learning)

This project implements an evolution algorithm that integrates the hard-thresholding (HT) operator into the well-known natural evolution strategies (NES) algorithm. For the paper, please check [this arixv link](https://arxiv.org/abs/2405.01615v1).

### Details

- Parallelism. We employ the simple <code>joblib</code> package, finding it much faster than using <code>ray</code> on a single node cluster. (Our single node machine has 512 cores.)
- Policy Network. We apply the NESHT algorithm to a single linear layer agent, as it's well established that the capacity of a single linear layer suffices for Mujoco tasks.
- Gaussian Noise. Our comparisons are conducted on noisy Mujoco environments, where the state is a combination of Gaussian noise and environment-provided observations.
- For any questions about the implementation, please don't hesitate to contact me.
