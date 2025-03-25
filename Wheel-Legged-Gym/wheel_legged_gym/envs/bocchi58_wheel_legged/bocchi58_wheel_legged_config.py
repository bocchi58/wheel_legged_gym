from wheel_legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)

class Bocchi58WheelLeggedCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_observations = 27 #lin_vel(3)+ang_vel(3)+cmd(3)+dof_pos(6)+dof_vel(6)+action(6)
        num_privileged_obs = (
            LeggedRobotCfg.env.num_envs+ 7 * 11 + 3 + 6 * 7 + 3 + 3
        )

    class init_state(LeggedRobotCfg.init_state):
        pos = [0,0,0.25] #x,y,z [m]
        default_joint_angles = { # target angles when action = 0.0
            "lf0_joint": 0.1,
            "lf1_joint": -0.98,
            "l_wheel_joint": 0.0,
            "rf0_joint": -0.1,
            "rf1_joint": 0.98,
            "rwheel_joint": 0.0,
        }
    #缩放因子scale,需不需要修改，有什么作用
    #应该是为了归一化用的，一般不需要修改
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
        stiffness = {"f0":20.0,"f1":20.0,"wheel":0}  # [N*m/rad]                        #刚度 P
        damping = damping = {"f0": 1.0, "f1": 1.0, "wheel": 0.1}  # [N*m*s/rad]         #阻尼 D
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

    class sim:
        dt = 0.005 #控制频率为200hz
        substeps = 1
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = (
                2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            )


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
    #命令
    class commands:
        curriculum = True
        basic_max_curriculum = 2.5
        advanced_max_curriculum = 1.5
        curriculum_threshold = 0.7
        num_commands = 3  # lin_vel_x,, ang_vel_yaw, height,heading (in heading mode ang_vel_yaw is recomputed from heading error) jump
        resampling_time = 5.0  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            #在课程学习里还会进一步修改
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-3.14, 3.14]  # min max [rad/s]
            height = [0.25, 0.45]
            heading = [-3.14, 3.14]
            #增加跳跃
            # jump = [0,1.0]

    class rewards:
        class scales:
            tracking_lin_vel = 1.0
            tracking_lin_vel_enhance = 1.0
            tracking_ang_vel = 1.0

            base_height = 1.0
            nominal_state = -0.1
            lin_vel_z = -2.0
            ang_vel_xy = -0.05  
            orientation = -20.0  #姿态控制，主要是控制roll轴，保持roll轴为0

            dof_vel = -5e-5
            dof_acc = -2.5e-7
            torques = -0.0001
            action_rate = -0.01
            action_smooth = -0.01

            collision = -1.0
            dof_pos_limits = -1.0

            spilts = -1.0 #防止劈叉
            control_theta = -1.0 #控制theta角度尽量为0


        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        clip_single_reward = 1
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = (
            0.97  # percentage of urdf limits, values above this limit are penalized
        )
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        base_height_target = 0.18
        max_contact_force = 100.0  # forces above this value are penalized

    #训练策略
class Bocchi58WheelLeggedCfgPPO(LeggedRobotCfgPPO):
    class policy:
        #代表的是探索能力
        # init_noise_std = 0.5
        init_noise_std = 1
        #网络大小尽量不要超过以下部分，否则将很难在32上面部署
        actor__dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        encoder_hidden_dims = [64, 16]
        #这里改成使用"lrelu""
        activation = "lrelu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for ActorCriticSequence
        num_encoder_obs = (
            LeggedRobotCfg.env.obs_history_length * LeggedRobotCfg.env.num_observations
        )
        latent_dim = 10  # at least 3 to estimate base linear velocity

    class algorithm(LeggedRobotCfgPPO.algorithm):
        kl_decay = (
            LeggedRobotCfgPPO.algorithm.desired_kl - 0.002
        ) / LeggedRobotCfgPPO.runner.max_iterations

    class runner(LeggedRobotCfgPPO.runner):
        # logging
        experiment_name = "bocchi58_wheel_legged"