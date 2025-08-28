using MDMA
using SubmodularMaximization


# Modify experiment_name for different experiments
experiment_name = "split_and_join"
path_to_experiments = "./experiments"
starting_states = Dict()


# Modified init_discrete_problem from MDMA_Experiment
cutoff = 100.0
sensor = MDMA.PinholeCameraModel([4.4, 4.4], [1920.0, 1080.0], [6.46, 3.64], 0.0, 0.0, 3.0)
move_dist = 0

multi_configs = configs_from_file(
    "$(path_to_experiments)/$(experiment_name)/$(experiment_name)_data.json",
    experiment_name,
    move_dist,
)
multi_configs.sensor = sensor

# Set cameras around the grid
starting_states[experiment_name] = [
    MDMA.MDPState(UAVState(15, 2, Symbol(:S)), multi_configs.horizon),
    MDMA.MDPState(UAVState(2, 15, Symbol(:E)), multi_configs.horizon),
    MDMA.MDPState(UAVState(28, 15, Symbol(:W)), multi_configs.horizon),
    MDMA.MDPState(UAVState(6, 28, Symbol(:N)), multi_configs.horizon),
    MDMA.MDPState(UAVState(24, 28,Symbol(:N)), multi_configs.horizon)
]

robot_states = starting_states[experiment_name]

problem = MDMA.MultiRobotTargetCoverageProblem(robot_states, multi_configs)


# Modified run_experiment from MDMA_Experiment
# Solving output
println("Solving Solution for $(experiment_name) using MultipleRoundsGreedyPlanner")
println("Num Robots", problem.configs.num_robots)
solution =  solve_sequential_multiround(problem, problem.configs.num_robots) # Change for different solver

# Save the solution
MDMA.save_solution(
    experiment_name,
    path_to_experiments,
    "multipleroundsgreedy", # Change for different solver
    solution,
    problem.configs,
)