"""
Solves the basic IK problem.
"""

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk


def solve_ik(
    robot: pk.Robot,
    target_link_name: str,
    target_wxyz: onp.ndarray,
    target_position: onp.ndarray,
) -> onp.ndarray:
    """
    Solves the basic IK problem for a robot.

    Args:
        robot: PyRoKi Robot.
        target_link_name: String name of the link to be controlled.
        target_wxyz: onp.ndarray. Target orientation.
        target_position: onp.ndarray. Target position.

    Returns:
        cfg: onp.ndarray. Shape: (robot.joint.actuated_count,).
    # """
    # print(f"🧮 IK SOLVE INPUT - Target link: {target_link_name}")
    # print(f"📍 IK SOLVE INPUT - Target position: {target_position}")
    # print(f"🔄 IK SOLVE INPUT - Target orientation (wxyz): {target_wxyz}")
    
    assert target_position.shape == (3,) and target_wxyz.shape == (4,)
    # print(robot.links.names)
    target_link_index = robot.links.names.index(target_link_name)
    # print(f"🔗 IK SOLVE - Target link index: {target_link_index}")
    
    cfg = _solve_ik_jax(
        robot,
        jnp.array(target_link_index),
        jnp.array(target_wxyz),
        jnp.array(target_position),
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)
    
    # print(f"⚙️ IK SOLVE OUTPUT - Joint configuration (rad): {onp.array(cfg)}")
    # print(f"📐 IK SOLVE OUTPUT - Joint configuration (deg): {onp.rad2deg(onp.array(cfg))}")
    
    return onp.array(cfg)


@jdc.jit
def _solve_ik_jax(
    robot: pk.Robot,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
) -> jax.Array:
    joint_var = robot.joint_var_cls(0)
    factors = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(target_wxyz), target_position),
            target_link_index,
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        pk.costs.limit_cost(
            robot,
            joint_var,
            weight=100.0,
        ),
        pk.costs.smoothness_cost(
            robot.joint_var_cls(0),
            robot.joint_var_cls(1),
            jnp.array([0.2])[None],
        ),
        # pk.costs.five_point_velocity_cost(
        #     robot,
        #     robot.joint_var_cls(0),
        #     robot.joint_var_cls(1),
        #     robot.joint_var_cls(3),
        #     robot.joint_var_cls(4),
        #     dt,
        #     jnp.array([10.0])[None],
        # ),
        # pk.costs.five_point_acceleration_cost(
        #     robot.joint_var_cls(2),
        #     robot.joint_var_cls(0),
        #     robot.joint_var_cls(1),
        #     robot.joint_var_cls(3),
        #     robot.joint_var_cls(4),
        #     dt,
        #     jnp.array([10.0])[None],
        # ),
        # pk.costs.five_point_jerk_cost(
        #     robot.joint_var_cls(0),
        #     robot.joint_var_cls(1),
        #     robot.joint_var_cls(2),
        #     robot.joint_var_cls(4),
        #     robot.joint_var_cls(5),
        #     robot.joint_var_cls(6),
        #     dt,
        #     jnp.array([0.1])[None],
        # ),
    ]
    sol = (
        jaxls.LeastSquaresProblem(factors, [joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
    )
    return sol[joint_var]
