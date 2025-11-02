# MPPI Controller Critics - Mathematical Principles

This directory contains the critic functions used in the Model Predictive Path Integral (MPPI) controller for navigation. Each critic evaluates different aspects of generated trajectories to compute a cost function that guides the selection of optimal control inputs.

## Overview

MPPI generates multiple candidate trajectories by sampling control inputs and evaluates them using a composite cost function:

\[ J(\mathbf{u}) = \sum_{t=0}^{T-1} \left[ \sum_{i=1}^{N} w_i \cdot c_i(\mathbf{x}_t, \mathbf{u}_t) \right] \]

where:
- \(\mathbf{u}\) is the control sequence
- \(\mathbf{x}_t\) is the state at time \(t\)
- \(c_i\) are individual critic costs
- \(w_i\) are critic weights
- \(N\) is the number of critics

Each critic below contributes a term \(c_i\) to this cost function.

## Individual Critics

### 1. ConstraintCritic

**Purpose**: Enforces velocity and turning constraints to ensure physically feasible trajectories.

**Mathematical Principle**:
\[ c_{\text{constraint}} = \left[ \sum_{t=0}^{T-1} \left( \max(v_t - v_{\max}, 0) + \max(v_{\min} - v_t, 0) + \max\left(\frac{r_{\min}}{|v_x|} - |\omega|, 0\right) \right) \cdot \Delta t \cdot w \right]^p \]

where:
- \(v_t = \sqrt{v_x^2 + v_y^2}\) is the total velocity magnitude
- \(v_{\max}, v_{\min}\) are maximum and minimum allowed velocities
- \(r_{\min}\) is the minimum turning radius
- \(\omega\) is the angular velocity
- \(w\) is the weight, \(p\) is the power

### 2. CostCritic

**Purpose**: Avoids obstacles using costmap information, with special handling for collision detection.

**Mathematical Principle**:
\[ c_{\text{cost}} = \left[ \frac{w}{254} \cdot \frac{1}{T} \sum_{t=0}^{T-1} \begin{cases} 
c_{\text{collision}} & \text{if collision detected} \\
c_{\text{critical}} & \text{if inscribed obstacle} \\
c_{\text{costmap}}(x_t, y_t) & \text{otherwise}
\end{cases} \right]^p \]

where:
- \(c_{\text{costmap}}\) is the costmap value at position \((x_t, y_t)\)
- Costs are normalized by 254 (maximum costmap value)

### 3. GoalAngleCritic

**Purpose**: When near the goal, penalizes deviation from the goal orientation.

**Mathematical Principle** (applied when within threshold distance):
\[ c_{\text{goal\_angle}} = \left[ w \cdot \frac{1}{T} \sum_{t=0}^{T-1} |\theta_{\text{shortest}}(\theta_t, \theta_{\text{goal}})| \right]^p \]

where:
- \(\theta_t\) is the trajectory yaw at time \(t\)
- \(\theta_{\text{goal}}\) is the goal yaw
- \(\theta_{\text{shortest}}\) is the shortest angular distance

### 4. GoalCritic

**Purpose**: Drives the trajectory toward the goal position.

**Mathematical Principle** (applied when within threshold distance):
\[ c_{\text{goal}} = \left[ w \cdot \frac{1}{T} \sum_{t=0}^{T-1} \sqrt{(x_t - x_{\text{goal}})^2 + (y_t - y_{\text{goal}})^2} \right]^p \]

### 5. ObstaclesCritic

**Purpose**: Provides repulsive forces from obstacles based on distance calculations.

**Mathematical Principle**:
\[ c_{\text{obstacles}} = \left[ w_{\text{critical}} \cdot c_{\text{raw}} + w_{\text{repulsion}} \cdot \frac{1}{T} \sum_{t=0}^{T-1} \max(r_{\text{inflation}} - d_t, 0) \right]^p \]

where:
- \(d_t\) is the distance to nearest obstacle
- \(r_{\text{inflation}}\) is the inflation radius
- \(c_{\text{raw}}\) is collision cost if collision detected

Distance to obstacle is calculated from costmap using:
\[ d = r_{\text{inscribed}} - \frac{\ln(c_{\text{costmap}}) - \ln(253)}{s} \]

where \(s\) is the scaling factor.

### 6. PathAlignCritic

**Purpose**: Keeps trajectories aligned with the reference path.

**Mathematical Principle**:
\[ c_{\text{align}} = \left[ w \cdot \frac{1}{N_{\text{samples}}} \sum_{i=1}^{N_{\text{samples}}} \sqrt{(x_i - x_{\text{path}}(s_i))^2 + (y_i - y_{\text{path}}(s_i))^2 + k \cdot \theta_{\text{shortest}}^2(\theta_i, \theta_{\text{path}}(s_i))} \right]^p \]

where:
- \(s_i\) is the path parameter corresponding to trajectory point \(i\)
- \(k = 1\) if using path orientations, \(k = 0\) otherwise

### 7. PathAlignLegacyCritic

**Purpose**: Legacy version of path alignment using segment-based distance calculation.

**Mathematical Principle**:
Similar to PathAlignCritic but finds minimum distance to path segments rather than interpolated points.

### 8. PathAngleCritic

**Purpose**: Penalizes angular deviation from the path direction.

**Mathematical Principle**:
\[ c_{\text{path\_angle}} = \left[ w \cdot \frac{1}{T} \sum_{t=0}^{T-1} |\theta_{\text{shortest}}(\theta_t, \theta_{\text{path}}(x_t, y_t))| \right]^p \]

where \(\theta_{\text{path}}(x_t, y_t)\) is the angle to the path point ahead.

### 9. PathFollowCritic

**Purpose**: Ensures trajectory endpoints stay close to the path.

**Mathematical Principle**:
\[ c_{\text{follow}} = \left[ w \cdot \sqrt{(x_T - x_{\text{path}})^2 + (y_T - y_{\text{path}})^2} \right]^p \]

where \((x_T, y_T)\) is the trajectory endpoint.

### 10. PreferForwardCritic

**Purpose**: Encourages forward motion over backward motion.

**Mathematical Principle**:
\[ c_{\text{forward}} = \left[ w \cdot \sum_{t=0}^{T-1} \max(-v_{x,t}, 0) \cdot \Delta t \right]^p \]

### 11. TwirlingCritic

**Purpose**: Penalizes excessive rotation to prevent spinning in place.

**Mathematical Principle**:
\[ c_{\text{twirl}} = \left[ w \cdot \frac{1}{T} \sum_{t=0}^{T-1} |\omega_t| \right]^p \]

### 12. VelocityDeadbandCritic

**Purpose**: Penalizes velocities below a minimum threshold to encourage active motion.

**Mathematical Principle**:
\[ c_{\text{deadband}} = \left[ w \cdot \sum_{t=0}^{T-1} \sum_{i=0}^{2} \max(v_{\text{deadband},i} - |v_{i,t}|, 0) \cdot \Delta t \right]^p \]

where \(v_0 = v_x\), \(v_1 = v_y\) (if holonomic), \(v_2 = \omega\).

## CriticData Structure

CriticData 结构体是 MPPI 控制器中用于传递给各个 critic 函数的数据结构。它封装了评估轨迹成本所需的所有关键信息。

### 结构体概述

CriticData 作为数据容器，在 MPPI 控制器的优化过程中传递给所有 critic 函数。这些 critic 函数使用这些数据来计算轨迹的成本，从而选择最优的控制序列。

### 成员变量详解

#### 核心状态和轨迹数据
- **`const models::State & state`**: 当前机器人状态，包含位置、速度、朝向等信息。这是 critic 评估的基础状态。
- **`const models::Trajectories & trajectories`**: MPPI 生成的候选轨迹集合。每个轨迹包含一系列状态点和控制输入。
- **`const models::Path & path`**: 全局规划路径，用于路径跟随相关的 critic。
- **`const geometry_msgs::msg::Pose & goal`**: 最终目标位姿，用于目标导向的 critic。

#### 成本计算相关
- **`xt::xtensor<float, 1> & costs`**: 存储每个轨迹的总成本值的一维数组。这是 critic 函数的主要输出，每个 critic 都会累加到这个数组中。
- **`float & model_dt`**: 模型的时间步长，用于时间相关的成本计算（如积分）。

#### 控制和状态标志
- **`bool fail_flag`**: 失败标志，如果某个 critic 检测到严重问题（如碰撞），可以设置此标志来标记轨迹无效。
- **`nav2_core::GoalChecker * goal_checker`**: 目标检查器，用于判断是否达到目标。
- **`std::shared_ptr<MotionModel> motion_model`**: 运动模型指针，用于轨迹生成和运动学约束检查。

#### 路径相关优化
- **`std::optional<std::vector<bool>> path_pts_valid`**: 可选的路径点有效性向量，标记路径上的哪些点仍然有效（未被障碍物阻塞）。
- **`std::optional<size_t> furthest_reached_path_point`**: 可选的索引，指示轨迹能到达的最远路径点，用于路径跟随评估。

### 使用方式

在 MPPI 优化循环中：
1. 生成候选轨迹
2. 为每个 critic 创建 CriticData 实例
3. 依次调用各个 critic 函数，每个函数读取相关数据并累加成本到 `costs` 数组
4. 根据总成本选择最优轨迹

这种设计使得 critic 函数模块化，每个 critic 只关注特定的评估方面（如障碍物回避、路径跟随等），而共享的数据通过这个结构体高效传递。

## Common Parameters

Most critics share these parameters:
- `cost_power` (\(p\)): Power to which the cost is raised (typically 1)
- `cost_weight` (\(w\)): Weight multiplier for the critic
- `threshold_to_consider`: Distance threshold for applying the critic

## Implementation Notes

- All costs are computed using xtensor for efficient vectorized operations
- Costs are accumulated in `data.costs` as xtensor arrays
- Collision detection uses costmap values and footprint checking
- Angular distances use shortest angular difference to handle wraparound
- Distance calculations use Euclidean metrics unless otherwise specified