from wheel_legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)

class Bocchi58WheelLeggedCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_observations = 21 #lin_vel(3)+ang_vel(3)+cmd(3)+6个状态量+action(6)
        num_privileged_obs = (
            LeggedRobotCfg.env.num_envs+ 7 * 11 + 3 + 6 * 7 + 3 + 3
        )

    class init_state(LeggedRobotCfg.init_state):
        pos = [0,0,0.25] #x,y,z [m]
        default_joint_angles = { # target angles when action = 0.0
            "lf0_Joint": 0.1,
            "lf1_Joint": -0.98,
            "l_wheel_Joint": 0.0,
            "rf0_Joint": -0.1,
            "rf1_Joint": 0.98,
            "r_wheel_Joint": 0.0,
        }
    #缩放因子scale,需不需要修改，有什么作用
    class control(LeggedRobotCfg.control):
        #这里的scale是什么意思
        pos_action_scale = 0.5
        vel_action_scale = 10.0


        l0_offset = 0
        feedforward_force = 100 #[N]前馈力

        kp_theta = 50.0  # [N*m/rad]
        kd_theta = 3.0  # [N*m*s/rad]
        kp_l0 = 900.0  # [N/m]
        kd_l0 = 20.0  # [N*s/m]

        #什么含义，需要修改吗
        # PD Drive parameters:
        stiffness = {"f0":40.0,"f1":40.0,"wheel":0}  # [N*m/rad]
        damping = damping = {"f0": 1.0, "f1": 1.0, "wheel": 0.5}  # [N*m*s/rad]
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 2
    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            dof_acc = 0.0025
            height_measurements = 5.0
            torque = 0.05

            clip_observations = 100.0
            clip_actions = 100.0
            
            l0 = 5.0
            l0_dot = 0.25

    class noise(LeggedRobotCfg.noise):
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            l0 = 0.02
            l0_dot = 0.1

    class asset(LeggedRobotCfg.asset):
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/balance/urdf/balance.urdf"
        name = "WheelLegged"
        offset = 0
        l1 = 0.215  #大腿
        l2 = 0.251  #小腿
        penalize_contacts_on = ["lf", "rf", "base"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
    #训练策略
class Bocchi58WheelLeggedCfgPPO(LeggedRobotCfgPPO):
    class policy:
        init_noise_std = 0.5
        actor__dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

        # only for ActorCriticSequence
        num_encoder_obs = (
            LeggedRobotCfg.env.obs_history_length * LeggedRobotCfg.env.num_observations
        )
        latent_dim = 10  # at least 3 to estimate base linear velocity
        encoder_hidden_dims = [64, 16]


    class algorithm(LeggedRobotCfgPPO.algorithm):
        kl_decay = (
            LeggedRobotCfgPPO.algorithm.desired_kl - 0.002
        ) / LeggedRobotCfgPPO.runner.max_iterations

    class runner(LeggedRobotCfgPPO.runner):
        # logging
        experiment_name = "boochi58_wheel_legged"