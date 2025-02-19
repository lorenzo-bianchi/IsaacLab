# env_cfg
viewer: {'eye': (7.5, 7.5, 7.5), 'lookat': (0.0, 0.0, 0.0), 'cam_prim_path': '/OmniverseKit_Persp', 'resolution': (1280, 720), 'origin_type': 'world', 'env_index': 0, 'asset_name': None, 'body_name': None}
sim: {'physics_prim_path': '/physicsScene', 'device': 'cpu', 'dt': 0.01, 'render_interval': 2, 'gravity': (0.0, 0.0, -9.81), 'enable_scene_query_support': True, 'use_fabric': True, 'disable_contact_processing': True, 'physx': {'solver_type': 1, 'min_position_iteration_count': 1, 'max_position_iteration_count': 255, 'min_velocity_iteration_count': 0, 'max_velocity_iteration_count': 255, 'enable_ccd': False, 'enable_stabilization': True, 'enable_enhanced_determinism': False, 'bounce_threshold_velocity': 0.5, 'friction_offset_threshold': 0.04, 'friction_correlation_distance': 0.025, 'gpu_max_rigid_contact_count': 8388608, 'gpu_max_rigid_patch_count': 163840, 'gpu_found_lost_pairs_capacity': 2097152, 'gpu_found_lost_aggregate_pairs_capacity': 33554432, 'gpu_total_aggregate_pairs_capacity': 2097152, 'gpu_collision_stack_size': 67108864, 'gpu_heap_capacity': 67108864, 'gpu_temp_buffer_capacity': 16777216, 'gpu_max_num_partitions': 8, 'gpu_max_soft_body_contacts': 1048576, 'gpu_max_particle_contacts': 1048576}, 'physics_material': {'func': 'omni.isaac.lab.sim.spawners.materials.physics_materials:spawn_rigid_body_material', 'static_friction': 1.0, 'dynamic_friction': 1.0, 'restitution': 0.0, 'improve_patch_friction': True, 'friction_combine_mode': 'multiply', 'restitution_combine_mode': 'multiply', 'compliant_contact_stiffness': 0.0, 'compliant_contact_damping': 0.0}, 'render': {'enable_translucency': False, 'enable_reflections': False, 'enable_global_illumination': False, 'antialiasing_mode': 'DLSS', 'enable_dlssg': False, 'dlss_mode': 0, 'enable_direct_lighting': True, 'samples_per_pixel': 1, 'enable_shadows': True, 'enable_ambient_occlusion': False}}
ui_window_class_type: omni.isaac.lab_tasks.direct.quadcopter.quadcopter_env:QuadcopterEnvWindow
seed: None
decimation: 2
is_finite_horizon: False
episode_length_s: 20.0
scene: {'num_envs': 1, 'env_spacing': 2.5, 'lazy_sensor_update': True, 'replicate_physics': True}
events: None
observation_space: 23
num_observations: None
state_space: 0
num_states: None
observation_noise_model: None
action_space: 4
num_actions: None
action_noise_model: None
rerender_on_reset: False
robot: {'class_type': 'omni.isaac.lab.assets.articulation.articulation:Articulation', 'prim_path': '/World/envs/env_.*/Robot', 'spawn': {'func': 'omni.isaac.lab.sim.spawners.from_files.from_files:spawn_from_usd', 'visible': True, 'semantic_tags': None, 'copy_from_source': False, 'mass_props': None, 'deformable_props': None, 'rigid_props': {'rigid_body_enabled': None, 'kinematic_enabled': None, 'disable_gravity': False, 'linear_damping': None, 'angular_damping': None, 'max_linear_velocity': None, 'max_angular_velocity': None, 'max_depenetration_velocity': 10.0, 'max_contact_impulse': None, 'enable_gyroscopic_forces': True, 'retain_accelerations': None, 'solver_position_iteration_count': None, 'solver_velocity_iteration_count': None, 'sleep_threshold': None, 'stabilization_threshold': None}, 'collision_props': None, 'activate_contact_sensors': False, 'scale': None, 'articulation_props': {'articulation_enabled': None, 'enabled_self_collisions': False, 'solver_position_iteration_count': 4, 'solver_velocity_iteration_count': 0, 'sleep_threshold': 0.005, 'stabilization_threshold': 0.001, 'fix_root_link': None}, 'fixed_tendons_props': None, 'joint_drive_props': None, 'visual_material_path': 'material', 'visual_material': None, 'usd_path': 'http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/Crazyflie/cf2x.usd', 'variants': None}, 'init_state': {'pos': (0.0, 0.0, 0.5), 'rot': (1.0, 0.0, 0.0, 0.0), 'lin_vel': (0.0, 0.0, 0.0), 'ang_vel': (0.0, 0.0, 0.0), 'joint_pos': {'.*': 0.0}, 'joint_vel': {'m1_joint': 200.0, 'm2_joint': -200.0, 'm3_joint': 200.0, 'm4_joint': -200.0}}, 'collision_group': 0, 'debug_vis': False, 'soft_joint_pos_limit_factor': 1.0, 'actuators': {'dummy': {'class_type': 'omni.isaac.lab.actuators.actuator_pd:ImplicitActuator', 'joint_names_expr': ['.*'], 'effort_limit': None, 'velocity_limit': None, 'stiffness': 0.0, 'damping': 0.0, 'armature': None, 'friction': None}}}
debug_vis: True
terrain: {'class_type': 'omni.isaac.lab.terrains.terrain_importer:TerrainImporter', 'collision_group': -1, 'prim_path': '/World/ground', 'num_envs': 1, 'terrain_type': 'plane', 'terrain_generator': None, 'usd_path': None, 'env_spacing': 2.5, 'visual_material': {'func': 'omni.isaac.lab.sim.spawners.materials.visual_materials:spawn_preview_surface', 'diffuse_color': (0.065, 0.0725, 0.08), 'emissive_color': (0.0, 0.0, 0.0), 'roughness': 0.5, 'metallic': 0.0, 'opacity': 1.0}, 'physics_material': {'func': 'omni.isaac.lab.sim.spawners.materials.physics_materials:spawn_rigid_body_material', 'static_friction': 1.0, 'dynamic_friction': 1.0, 'restitution': 0.0, 'improve_patch_friction': True, 'friction_combine_mode': 'multiply', 'restitution_combine_mode': 'multiply', 'compliant_contact_stiffness': 0.0, 'compliant_contact_damping': 0.0}, 'max_init_terrain_level': None, 'debug_vis': False}
thrust_to_weight: 1.9
moment_scale: 0.01
lin_vel_reward_scale: -0.2
ang_vel_reward_scale: -0.05
approaching_goal_reward_scale: 900.0
convergence_goal_reward_scale: 600.0
yaw_reward_scale: 300.0
new_goal_reward_scale: 100.0
cmd_smoothness_reward_scale: -1.5
cmd_body_rates_reward_scale: -0.3
death_cost: -1000.0

# agent_cfg
seed: 42
device: cuda:0
num_steps_per_env: 24
max_iterations: 200
empirical_normalization: False
policy: {'class_name': 'ActorCritic', 'init_noise_std': 1.0, 'actor_hidden_dims': [64, 64], 'critic_hidden_dims': [64, 64], 'activation': 'elu'}
algorithm: {'class_name': 'PPO', 'value_loss_coef': 1.0, 'use_clipped_value_loss': True, 'clip_param': 0.2, 'entropy_coef': 0.0, 'num_learning_epochs': 5, 'num_mini_batches': 4, 'learning_rate': 0.0005, 'schedule': 'adaptive', 'gamma': 0.99, 'lam': 0.95, 'desired_kl': 0.01, 'max_grad_norm': 1.0}
save_interval: 50
experiment_name: quadcopter_direct
run_name: 
logger: tensorboard
neptune_project: isaaclab
wandb_project: isaaclab
resume: False
load_run: 2025-02-11_16-18-00
load_checkpoint: model_.*.pt

# args_cli
video: False
video_length: 200
disable_fabric: False
num_envs: 1
task: Isaac-Quadcopter-Direct-v0
experiment_name: None
run_name: None
resume: None
load_run: 2025-02-11_16-18-00
checkpoint: None
logger: None
log_project_name: None
device: cpu
cpu: False
verbose: False
info: False
kit_args: 
headless: False
hide_ui: False
physics_gpu: 0
active_gpu: 0