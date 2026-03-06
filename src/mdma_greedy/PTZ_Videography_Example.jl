using MDMA
using SubmodularMaximization

# Modify experiment_name for different experiments
experiment_name = "split_and_join"
path_to_experiments = "./experiments"
experiment_data = "$(path_to_experiments)/$(experiment_name)/$(experiment_name)_data.json"
ptz_data = "$(path_to_experiments)/$(experiment_name)/$(experiment_name)_ptz_data.json"

# Set up and run GreedyPlanner
multi_configs = configs_from_file(experiment_data, experiment_name, ptz_data, 3.0)
starting_states = init_ptz_states(ptz_data, multi_configs.grid)
problem = MDMA.MultiRobotTargetCoverageProblem(starting_states, multi_configs)
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