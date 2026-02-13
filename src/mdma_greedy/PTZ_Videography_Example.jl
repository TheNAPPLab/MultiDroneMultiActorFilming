using MDMA
using SubmodularMaximization

# Modify experiment_name for different experiments
experiment_name = "split_and_join"
path_to_experiments = "./experiments"
starting_states = Dict()
pan_divisions = 8 # Number of pan angles to use
tilt_divisions = 5 # Number of tilt angles to use
zoom_divisions = 3 # Number of zoom values to use

# Modified init_discrete_problem from MDMA_Experiment
sensor = MDMA.PinholeCameraModel(4.4, [1920.0, 1080.0], [5.60, 3.15], 0.0, 3.0) # Initialize at 1x zoom
move_dist = 0
camera_positions = [
    (15.0, 2.0, 5.0),
    (2.0, 15.0, 5.0),
    (28.0, 15.0, 5.0),
    (6.0, 28.0, 5.0),
    (24.0, 28.0, 5.0)
]

# Set up and run GreedyPlanner
multi_configs = configs_from_file(
    "$(path_to_experiments)/$(experiment_name)/$(experiment_name)_data.json",
    experiment_name,
    move_dist,
    camera_positions,
    pan_divisions,
    tilt_divisions,
    zoom_divisions
)
multi_configs.sensor = sensor
starting_states[experiment_name] = [
    MDMA.MDPState(PTZState(15.0, 2.0, 5.0, 0.0, 0.0, 1.0), multi_configs.horizon),
    MDMA.MDPState(PTZState(2.0, 15.0, 5.0, 0.0, 0.0, 1.0), multi_configs.horizon),
    MDMA.MDPState(PTZState(28.0, 15.0, 5.0, 0.0, 0.0, 1.0), multi_configs.horizon),
    MDMA.MDPState(PTZState(6.0, 28.0, 5.0, 0.0, 0.0, 1.0), multi_configs.horizon),
    MDMA.MDPState(PTZState(24.0, 28.0, 5.0, 0.0, 0.0, 1.0), multi_configs.horizon)
]
robot_states = starting_states[experiment_name]
problem = MDMA.MultiRobotTargetCoverageProblem(robot_states, multi_configs)
println("Solving Solution for $(experiment_name) using GreedyPlanner")
solution = solve_sequential(problem)

# Save the solution
MDMA.save_solution(
    experiment_name,
    path_to_experiments,
    "greedy",
    solution,
    problem.configs,
)