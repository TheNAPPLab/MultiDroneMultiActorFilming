using MDMA
using SubmodularMaximization

function receding_horizon_planner(
    max_horizon::Int64, # Number of time steps in the horizon
    multi_configs::MultiDroneMultiActorConfigs,
    starting_states::Vector{MDMA.MDPState}
)::Solution
    # Initialize receding horizon loop variables
    final_solution = Solution(0.0, Tuple{Int64, Vector{MDMA.MDPState}}[])
    for i in 1:multi_configs.num_robots-1
        push!(final_solution.elements, (i, [starting_states[i]]))
    end
    temp_configs = deepcopy(multi_configs)
    temp_configs.grid = MDMA_Grid(multi_configs.grid, max_horizon)
    temp_configs.horizon = max_horizon

    # Receding horizon planner loop
    println("Horizon Size: $(max_horizon)")
    for t in 1:multi_configs.horizon-1
        # Determine time steps in the horizon
        t_end = min(t + max_horizon - 1, multi_configs.horizon)
        current_horizon = t_end - t + 1
        println("Current horizon t = $(t) to $(t_end)")
        
        # Decrease horizon size as end of data approaches
        if current_horizon != max_horizon
            temp_configs.grid = MDMA_Grid(temp_configs.grid, current_horizon)
            temp_configs.horizon = current_horizon
        end

        # Update known actor trajectories given horizon
        temp_configs.target_trajectories = multi_configs.target_trajectories[t:t_end, :]

        # Run current planner iteration
        problem = MDMA.MultiRobotTargetCoverageProblem(starting_states, temp_configs)
        current_solution = solve_sequential(problem)

        # Add the current time step's solution to the final solution
        for i in 1:multi_configs.num_robots-1
            current_solution.elements[i][2][2] = MDMA.MDPState(current_solution.elements[i][2][2], t+1, multi_configs.horizon)
            starting_states[i] = MDMA.MDPState(current_solution.elements[i][2][2].state, multi_configs.horizon)
            push!(final_solution.elements[i][2], current_solution.elements[i][2][2])
        end
        final_solution = Solution(final_solution.value + current_solution.value, final_solution.elements)
        println()
    end

    return final_solution
end

# Timed experiment using receding horizon planner
elapsed = @elapsed begin
    # Modify experiment_name for different experiments
    experiment_name = "split_and_join"
    path_to_experiments = "./experiments"
    experiment_data = "$(path_to_experiments)/$(experiment_name)/$(experiment_name)_data.json"
    ptz_data = "$(path_to_experiments)/$(experiment_name)/$(experiment_name)_ptz_data.json"
    horizon = 5
    
    # Set up and run GreedyPlanner using receding horizon planner
    multi_configs = configs_from_file(experiment_data, experiment_name, ptz_data, 3.0)
    starting_states = init_ptz_states(ptz_data, multi_configs.grid)
    println("Solving Solution for $(experiment_name) using GreedyPlanner")
    solution = receding_horizon_planner(horizon, multi_configs, starting_states)

    # Save the solution
    MDMA.save_solution(
        experiment_name,
        path_to_experiments,
        "greedy",
        solution,
        multi_configs,
    )
end

# Print timing results
minutes = Int(floor(elapsed / 60))
seconds = elapsed % 60
println("Runtime: $(minutes) min   $(seconds) sec")