# nav2_mppi_controller_py

This package is a from-scratch Python port of the Navigation2 Model Predictive
Path Integral (MPPI) controller.  It mirrors the behaviour of the C++ plugin in
`nav2_mppi_controller` while keeping the implementation lightweight and free of
ROS-specific dependencies.  All core components—noise sampling, optimiser,
critics, motion models, and controller interface—are implemented in NumPy and
work in standalone Python environments.

## 1. MPPI in a Nutshell

Model Predictive Path Integral control is a sampling-based stochastic optimal
control technique.  At every control cycle the algorithm:

1. Samples a batch of control sequences around a nominal sequence.
2. Rolls the robot dynamics forward under each control sequence, yielding a
   collection of trajectories.
3. Evaluates the trajectories with respect to user-defined cost (critic)
   functions.
4. Uses a soft-minimum weighting (derived from path integral theory) to build
   the next control command.

The method can be viewed as computing the expectation of control disturbances
under an exponential transformation of their accumulated cost.  This
transformation makes high-cost trajectories exponentially unlikely and
close-to-optimal trajectories dominate the integral.

## 2. Mathematical Foundations

MPPI solves a stochastic optimal control problem for a discrete-time nonlinear
system

\[
x_{k+1} = f(x_k, u_k) + w_k, \qquad w_k \sim \mathcal{N}(0, \Sigma)
\]

where \(x_k \in \mathbb{R}^n\) is the state, \(u_k \in \mathbb{R}^m\) is the
control action, and \(w_k\) is injected process noise.  Over a planning horizon
of \(T\) steps we search for the control sequence
\(U = \{u_0, u_1, \dots, u_{T-1}\}\) that minimises the objective

\[
J(U) = \phi(x_T) + \sum_{k=0}^{T-1} q(x_k, u_k)
\]

with \(\phi\) a terminal cost and \(q\) the running cost.  MPPI turns this
problem into a Monte-Carlo estimate of the path integral solution of the
stochastic Hamilton–Jacobi–Bellman equation.  Starting from a nominal control
sequence \(\bar{U}\), noises \(\epsilon^{(i)}\) are sampled and added to obtain
candidate controls \(U^{(i)} = \bar{U} + \epsilon^{(i)}\).  Each rollout
produces a trajectory \(\{x^{(i)}_k\}_{k=0}^{T}\) and an associated cost-to-go

\[S^{(i)} = \phi\!\left(x^{(i)}_T\right) + \sum_{k=0}^{T-1} \Big(q\!\left(x^{(i)}_k, u^{(i)}_k\right) + \tfrac{\gamma}{2} \left(\epsilon^{(i)}_k\right)^{\!\top} R^{-1} \epsilon^{(i)}_k\Big)\]

where \(R = \Sigma / \lambda\) plays the role of a control cost matrix,
\(\lambda > 0\) is the temperature parameter, and \(\gamma\) scales the control
penalty.  The optimal control increment applied at the current time step is the
weighted average

\[
\begin{aligned}
\delta u_0 &= \sum_{i=1}^{N} w_i \, \epsilon^{(i)}_0, \\
w_i &= \frac{\exp\!\left(-S^{(i)} / \lambda\right)}
{\sum_{j=1}^{N} \exp\!\left(-S^{(j)} / \lambda\right)}.
\end{aligned}
\]

Low-cost rollouts receive exponentially larger weights, steering the control
toward optimal behaviour while the sampling noise guarantees continued
exploration.

### 2.1 Discretisation and Shifting

The controller operates with a fixed model time step \(\Delta t\).  The running
cost integrates the continuous objective

\[
\int_{t_0}^{t_0 + T} q\big(x(t), u(t)\big) \, dt
\approx \sum_{k=0}^{T-1} q(x_k, u_k) \, \Delta t,
\]

which is precisely the quantity accumulated during the discrete rollout.  Many
implementations apply *control sequence shifting* so that information from the
previous solution is reused.  After executing the first command, the nominal
sequence \(\bar{u}_k\) is shifted one step ahead:

\[
\bar{u}_{k}^{\text{next}} =
\begin{cases}
\bar{u}_{k+1}, & 0 \leq k < T-1, \\
\bar{u}_{T-1}^{\text{new}}, & k = T-1,
\end{cases}
\]

where \(\bar{u}_{T-1}^{\text{new}}\) is typically reinitialised to zero or the
mean of the sampled noises.  This optional update often improves convergence
speed because it preserves useful information about recent optimisation steps.

### 2.2 Noise Sampling

The Python port maintains a Gaussian noise generator with configurable standard
deviations `(sigma_vx, sigma_vy, sigma_wz)` for the linear and angular
velocities.  These parameters shape the exploration radius in control space.
When the vehicle is non-holonomic, the `vy` component is set to zero.

## 3. Structure of the Python Package

The Python package mirrors the layout of the C++ controller:

| Module | Purpose |
| ------ | ------- |
| `models/*.py` | Tensor containers for controls, trajectories, paths, and optimiser settings. |
| `motion_models.py` | Diff-drive, Ackermann, and omnidirectional motion models with constraint projection. |
| `tools/noise_generator.py` | Gaussian noise sampling for MPPI rollouts (thread-safe). |
| `tools/utils.py` | Geometry helpers, angle utilities, smoothing filters, and convenience functions for building paths. |
| `critics/*.py` | Path-follow, goal-distance, goal-heading, and control-effort critic implementations. |
| `critic_manager.py` | Simple manager that evaluates all registered critics. |
| `optimizer.py` | Core MPPI optimiser producing the next control command from sampled trajectories. |
| `controller.py` | Thin wrapper that stores the active plan and exposes a `compute_velocity_command` call. |
| `tools/utils.py` | Re-implements the geometry and filtering helpers used throughout the algorithm. |
| `tools/noise_generator.py` | Generates the Gaussian disturbances for each rollout in a thread-safe manner. |
| `tools/parameters_handler.py` | Minimal dynamic-parameter helper used by the optimiser and noise generator. |

## 4. Example: Kinematic Car Demo

A runnable example is provided in `examples/car_demo.py`.  It instantiates the
diff-drive motion model, attaches the default critic set, and drives a robot
around a circular reference path using the Python MPPI controller.  An animated
trace of the trajectory is written to `media/python_demo.gif`.

```bash
python3 -m nav2_mppi_controller_py.examples.car_demo
```

Dependencies: `numpy`, `matplotlib`, and `pillow` (for GIF export).  The script
uses the Agg backend by default, so it runs headless on CLI systems.

## 5. Critic Design Overview

Each critic receives a `CriticData` instance containing:

- The sampled trajectories.
- The robot state and goal pose.
- The path in tensor form.
- A mutable cost array to accumulate penalties.
- Convenience flags such as `fail_flag` indicating whether optimisation should
  retry with a reset.

Critics express objectives such as obstacle avoidance, path following,
velocity deadbands, or goal alignment.  By keeping each objective isolated, the
developer can tune or replace individual behaviours without rewriting the core
controller.

## 6. Practical Considerations

### 6.1 ROS 2 Integration

The current Python package cannot yet plug into the Nav2 server stack because
several dependencies (`nav2_core::Controller`, costmap access, TF buffers) only
have C++ interfaces.  To complete the port you must either:

- Provide Python bindings to those C++ classes (for example with `pybind11`), or
- Re-architect the controller to communicate with Nav2 through custom ROS 2
  topics and services rather than pluginlib.

### 6.2 Performance

MPPI relies heavily on vectorised numerical operations.  The C++ implementation
uses `xtensor`, which maps well to Eigen-style performance.  A pure Python port
should rely on NumPy calls (already reflected in the current data structures)
and avoid Python loops whenever possible.  If additional speed is required,
consider translating the hot paths (rollouts, cost evaluations) to Cython,
Numba, or a compiled extension.

### 6.3 Numerical Stability

Key parameters to monitor:

- **Temperature (`lambda`)**: too small causes numerical underflow in the
  exponential weights; too large makes the optimisation sluggish.
- **Model time step (`dt`)**: must align with the controller frequency; otherwise
  shifting and rollout assumptions break down.
- **Sampling variance**: large variances explore widely but can violate motion
  constraints unless properly clamped by the motion model.
- **Retry logic**: the C++ algorithm retries optimisation when critics flag
  failures.  The Python port should replicate this logic to avoid oscillations.

## 7. Possible Extensions

1. **Nav2 integration.**  Add bindings that expose the Python controller as a
   pluginlib-compatible component or bridge the ROS 2 messaging layer.
2. **Additional critics.**  Recreate the full suite from the C++ controller
   (e.g., obstacle cost, constraint, velocity dead-band) or design new
   objectives tailored to specific robots.
3. **GPU sampling.**  Offload rollout generation to JAX, CuPy, or PyTorch for
   stronger real-time guarantees on embedded GPUs.
4. **Automated regression tests.**  Record target trajectories and validate cost
   reductions to detect behavioural regressions during tuning.

## 8. References

- Williams, Grady; Aldrich, Aaron; Theodorou, Evangelos. *Model Predictive Path
  Integral Control: Theory and Applications to Autonomous Driving*. 2016.
- Bhardwaj, Mohak et al. *Surreal: Open-Source Reinforcement Learning Control
  in Unstructured Environments.* ICRA 2022. (MPPI practical applications.)
- The original Nav2 MPPI controller documentation:
  <https://navigation.ros.org/plugins/index.html#mppi-controller>.

---

If you plan to continue the port, treat this README as a living document:
expand it with implementation notes, lessons learned while mapping C++ idioms
to Python, and benchmarks comparing the two controllers.  Contributions and
design discussions are welcome.
