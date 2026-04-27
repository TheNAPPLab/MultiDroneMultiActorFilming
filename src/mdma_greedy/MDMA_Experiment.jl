import JSON3
using Test
using SubmodularMaximization
using MDMA
using Base.Threads

using DataFrames
using CSV
using Plots
using Plots.PlotMeasures
using Statistics

export AssignmentPlanner, GreedyPlanner, FormationPlanner, MyopicPlanner, MultipleRoundsGreedyPlanner
export run_all_experiments, ExperimentsConfig
export blender_render_all_experiments, evaluate_all_experiments, evaluate_ptz_experiments, evaluate_ptz_actor_experiments

struct GreedyPlanner end
struct FormationPlanner end
struct AssignmentPlanner end
struct MyopicPlanner end
struct MultipleRoundsGreedyPlanner end

## Ensure that planners start in the same state for each experiment
planners = [FormationPlanner(), GreedyPlanner(), AssignmentPlanner(),MyopicPlanner(), MultipleRoundsGreedyPlanner()]
# planners = [AssignmentPlanner()]
# planners = [MyopicPlanner()]
#planners = [MultipleRoundsGreedyPlanner()]
planner_type_to_string = x -> lowercase(chop(string(x), head=5, tail=9))
planner_types = map(planner_type_to_string, planners)

# maps experiment names to starting states
starting_states = Dict()

function init_discrete_problem(
    experiment_name::String,
    path_to_experiments::String,
)

    # Set model parameters
    # focal_length = [20.0, 20.0]#mm
    # resolution = [3840.0, 2160.0]
    # lens_dim = [35.00, 24.00] #mm
    # pitch = -0.3490655 # if you change this also change animate_cameras.py
    cutoff = 100.0
    # sensor = MDMA.PinholeCameraModel(focal_length, resolution, lens_dim, 0.0, pitch, cutoff)
    sensor = MDMA.PinholeCameraModel([1574.89111, 1613.1925], [1920, 1080], [923.75228, 564.05564], 3.0)
    move_dist = 3

    multi_configs = configs_from_file(
        "$(path_to_experiments)/$(experiment_name)/$(experiment_name)_data.json",
        experiment_name,
        move_dist,
    )
    multi_configs.sensor = sensor

    if !haskey(starting_states, experiment_name)
        push!(starting_states, experiment_name => map(
            x -> MDMA.random_state(multi_configs.horizon, multi_configs.grid),
            1:multi_configs.num_robots,
        ))
    end

    robot_states = starting_states[experiment_name]

    multi_problem = MDMA.MultiRobotTargetCoverageProblem(robot_states, multi_configs)
    multi_problem

end

function run_experiment(
    experiment_name::String,
    path_to_experiments::String,
    _::MultipleRoundsGreedyPlanner,
)
    problem = init_discrete_problem(experiment_name, path_to_experiments)
    # Solving output
    println("Solving Solution for $(experiment_name) using MultipleRoundsGreedyPlanner")
    println("Num Robots", problem.configs.num_robots)
    solution =  solve_sequential_multiround(problem, problem.configs.num_robots)

    # Save the solution
    MDMA.save_solution(
        experiment_name,
        path_to_experiments,
        "multipleroundsgreedy",
        solution,
        problem.configs,
    )
end

function run_experiment(
    experiment_name::String,
    path_to_experiments::String,
    _::MyopicPlanner,
)

    problem = init_discrete_problem(experiment_name, path_to_experiments)
    # Solving output
    println("Solving Solution for $(experiment_name) using MyopicPlanner")
    solution = solve_myopic(problem)

    # Save the solution
    MDMA.save_solution(
        experiment_name,
        path_to_experiments,
        "myopic",
        solution,
        problem.configs,
    )
end

function run_experiment(
    experiment_name::String,
    path_to_experiments::String,
    _::GreedyPlanner,
)
    problem = init_discrete_problem(experiment_name, path_to_experiments)
    # Solving output
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
end

function run_experiment(
    experiment_name::String,
    path_to_experiments::String,
    _::AssignmentPlanner,
)
    
    cutoff = 100.0
    # sensor = MDMA.PinholeCameraModel(focal_length, resolution, lens_dim, 0.0, pitch, cutoff)
    sensor = MDMA.PinholeCameraModel([1574.89111, 1613.1925], [1920, 1080], [923.75228, 564.05564], 3.0)
    move_dist = 3

    multi_configs = configs_from_file(
        "$(path_to_experiments)/$(experiment_name)/$(experiment_name)_data.json",
        experiment_name,
        move_dist,
    )
    multi_configs.sensor = sensor

    if !haskey(starting_states, experiment_name)
        push!(starting_states, experiment_name => map(
            x -> MDMA.random_state(multi_configs.horizon, multi_configs.grid),
            1:multi_configs.num_robots,
        ))
    end

    robot_states = starting_states[experiment_name]

    problem = MDMA.MultiRobotTargetAssignmentProblem(robot_states, multi_configs)
    
    # Solving output
    println("Solving Solution for $(experiment_name) using AssignmentPlanner")
    solution = solve_sequential(problem)

    # Save the solution
    MDMA.save_solution(
        experiment_name,
        path_to_experiments,
        "assignment",
        solution,
        problem.configs,
    )
end

function run_experiment(
    experiment_name::String,
    path_to_experiments::String,
    _::FormationPlanner,
)

    # Not actually used by this planner
    # Just using the exiting function configs_from_file
    # To extract some information we need
    move_dist = 3
    configs = configs_from_file(
        "$(path_to_experiments)/$(experiment_name)/$(experiment_name)_data.json",
        experiment_name,
        move_dist,
    )

    # Initial States    
    # TODO: Random phase angle
    # Instead of starting states
    formation_configs = formation_from_multi(15.0, randn(), configs)

    println("Solving Solution for $(experiment_name) using FormationPlanner")
    solution = solve_formation(formation_configs)
    MDMA.save_solution(
        experiment_name,
        path_to_experiments,
        "formation",
        solution,
        configs,
    )
end

struct ExperimentsConfig
    path_to_experiments::String
    experiments::Vector{String}
end


ExperimentsConfig(path_to_experiments::String) = ExperimentsConfig(path_to_experiments, ["cluster", "cross_mix", "four_split","heavy_mixing","split_and_join", "spreadout_group", "track_runners", "priority_runners", "priority_speaker"])
# ExperimentsConfig(path_to_experiments::String) = ExperimentsConfig(path_to_experiments, ["split_and_join"])

# Experiment Steps
#  - Generate $(experiment_name)_data.json using a blender call
#  - Run the experiment
#  - Render images
#  - Evaluate the solution and save to two csv's
# Experiment List
# ["cross_mix", "cluster", "four_split", "spreadout_group"]
function run_all_experiments(config::ExperimentsConfig)
    # mtime is what we will store
    Threads.@threads for experiment in config.experiments
        path = "$(config.path_to_experiments)/$(experiment)"
        # Generate experiment data
        println("Building Experiment Data")
        data_cmd = `blender --background $(path)/$(experiment).blend  --python $(config.path_to_experiments)/export_experiment_data.py`
        run(data_cmd)

        # Run the experiment
        println("Running Experiment $(experiment)")
        planner_type_to_string = x -> lowercase(chop(string(x), head=0, tail=9))
        for planner in planners
            pl_string = planner_type_to_string(planner)
            println("Running planner $(pl_string) for $(experiment)")
            run_experiment(experiment, config.path_to_experiments, planner)
            println("Planner $(pl_string) complete for $(experiment)")
        end

        # Render Images
        println("Blender Rendering for $(experiment)")
        render_cmd = `blender --background $(path)/$(experiment).blend --python $(config.path_to_experiments)/animate_cameras.py`
        run(render_cmd)
        println("Render Finish for $(experiment)")

        # Evaluate Experiments
        evaluate_all_experiments(config)
        println("Experiment $(experiment) complete!")
    end
end

function blender_render_all_experiments(config::ExperimentsConfig)
    Threads.@threads for experiment in config.experiments
        path = "$(config.path_to_experiments)/$(experiment)"
        data_cmd = `blender --background $(path)/$(experiment).blend  --python $(config.path_to_experiments)/export_experiment_data.py`
        run(data_cmd)
        println("Blender Rendering for $(experiment)")
        render_cmd = `blender --background $(path)/$(experiment).blend  --python $(config.path_to_experiments)/animate_cameras.py`
        run(render_cmd)
        println("Render Finish for $(experiment)")
    end
end
function evaluate_all_experiments(config::ExperimentsConfig)
    # mtime is what we will store
    for experiment in config.experiments
        path = "$(config.path_to_experiments)/$(experiment)"

        println("Path $(path)")
        # Evaluate Experiments
        println("Evaluating PPA for $(experiment)")
        df_ppa = evaluate_solution(experiment, planner_types, config.path_to_experiments, PPAEvaluation())
        CSV.write("$(path)/ppa_evaluation.csv", df_ppa)
        save_plot(df_ppa, path, "PPA", experiment)

        println("Evaluating Image for $(experiment)")
        df_image = evaluate_solution(experiment, planner_types, config.path_to_experiments, ImageEvaluation())
        CSV.write("$(path)/image_evaluation.csv", df_image)
        save_plot(df_image, path, "Image", experiment)

        println("Experiment $(experiment) complete!")
    end
end
function save_plot(df::DataFrame, path::String, eval_kind::String, experiment_name::String)
    pl = plot()
    for planner in planner_types
        plot!(pl, df[!, :t], df[!, Symbol(planner)], label=planner)
    end
    xlabel!(pl,"TimeStep")
    ylabel!(pl, "Evaluation ($(eval_kind))")
    title!(pl, "$(eval_kind) Evaluation for $(experiment_name)")
    savefig(pl, "$(path)/$(lowercase(eval_kind))_evaluation.tex")
    savefig(pl, "$(path)/$(lowercase(eval_kind))_evaluation.png")
end

function evaluate_ptz_experiments(config::ExperimentsConfig, solution_filenames::Vector{String}, solution_labels::Vector{String}, study::String)
    for experiment in config.experiments
        path = "$(config.path_to_experiments)/$(experiment)"

        println("Path $(path)")
        # Evaluate Experiments
        println("Evaluating PPA for PTZ $(experiment)")
        df_ppa = evaluate_ptz_solution(experiment, solution_filenames, solution_labels, config.path_to_experiments, study, PPAEvaluation())
        CSV.write("$(path)/ppa_evaluation_$(study).csv", df_ppa)
        save_plot(df_ppa, path, "PPA", experiment, solution_labels, "Time Step", "PPA Objective", study, 18mm, 6mm)
        save_table(df_ppa, path, "ppa_evaluation_$(study)")

        println("Evaluating Scene Coverage for PTZ $(experiment)")
        df_ppa = evaluate_ptz_solution(experiment, solution_filenames, solution_labels, config.path_to_experiments, study, SceneCoverageEvaluation())
        CSV.write("$(path)/scene_coverage_evaluation_$(study).csv", df_ppa)
        save_plot(df_ppa, path, "Total Actors Covered", experiment, solution_labels, "Time Step", "Number of Actors", study, 6mm, 6mm)
        save_table(df_ppa, path, "scene_coverage_evaluation_$(study)")

        println("Experiment $(experiment) complete!")
    end
end

function evaluate_ptz_actor_experiments(config::ExperimentsConfig, solution_filenames::Vector{String}, solution_labels::Vector{String}, study::String)
    for experiment in config.experiments
        path = "$(config.path_to_experiments)/$(experiment)"
        for (i, solution_filename) in enumerate(solution_filenames)
            # Get label names for plots
            multi_configs =
                configs_from_file("$(path)/$(experiment)_data.json", experiment, "$(path)/$(experiment)_ptz_data_$(study).json", 1000.0)
            target_labels = String[]
            for i in axes(multi_configs.target_trajectories, 2)
                push!(target_labels, "Target $(i)")
            end

            println("Path $(path)")
            # Evaluate Experiments
            println("Evaluating Actor Coverage for PTZ $(experiment)")
            df_coverage = evaluate_ptz_solution(experiment, solution_filename, config.path_to_experiments, study, ActorCoverageEvaluation())
            CSV.write("$(path)/actor_coverage_evaluation_$(solution_labels[i])_$(study).csv", df_coverage)
            save_plot(df_coverage, path, "Actor Coverage $(solution_labels[i])", experiment, target_labels, "Time Step", "Number of Cameras", "$(solution_labels[i])_$(study)", 6mm, 6mm)
            save_table(df_coverage, path, "actor_coverage_evaluation_$(solution_labels[i])_$(study)")

            println("Evaluating Actor PPA for PTZ $(experiment)")
            df_ppa = evaluate_ptz_solution(experiment, solution_filename, config.path_to_experiments, study, ActorRewardEvaluation())
            CSV.write("$(path)/actor_reward_evaluation_$(solution_labels[i])_$(study).csv", df_ppa)
            save_plot(df_ppa, path, "Actor PPA", experiment, target_labels, "Time Step", "PPA Objective", "$(solution_labels[i])_$(study)", 12mm, 6mm)
            save_table(df_ppa, path, "actor_reward_evaluation_$(solution_labels[i])_$(study)")
        end
        println("Experiment $(experiment) complete!")
    end
end

function save_plot(df::DataFrame, path::String, eval_kind::String, experiment_name::String, solution_labels::Vector{String}, x_label::String, y_label::String, info::String, left_margin = 0mm, bottom_margin = 0mm)
    pl = plot(left_margin=left_margin, bottom_margin=bottom_margin)
    for sol in solution_labels
        plot!(pl, df[!, :t], df[!, sol], label=sol)
    end
    xlabel!(pl, "$(x_label)")
    ylabel!(pl, "$(y_label)")
    title!(pl, "$(eval_kind) Evaluation for $(experiment_name)")
    savefig(pl, "$(path)/$(lowercase(eval_kind))_evaluation_$(info).png")
    savefig(pl, "$(path)/$(lowercase(eval_kind))_evaluation_$(info).pdf")
end

function save_plot(df::DataFrame, path::String, eval_kind::String, experiment_name::String, solution_labels::Vector{String}, info::String, left_margin = 0mm, bottom_margin = 0mm)
    pl = plot(left_margin=left_margin, bottom_margin=bottom_margin)
    for sol in solution_labels
        plot!(pl, df[!, :t], df[!, sol], label=sol)
    end
    xlabel!(pl,"TimeStep")
    ylabel!(pl, "Evaluation ($(eval_kind))")
    title!(pl, "$(eval_kind) Evaluation for $(experiment_name)")
    savefig(pl, "$(path)/$(lowercase(eval_kind))_evaluation_$(info).png")
    savefig(pl, "$(path)/$(lowercase(eval_kind))_evaluation_$(info).pdf")
end

function save_table(df::DataFrame, path::String, eval_kind::String)
    data_cols = df[:, 2:end] # Discard time step column
    df_results = DataFrame(
        Actor = names(data_cols),
        Mean = mean.(eachcol(data_cols)),
        Median = median.(eachcol(data_cols)),
        SD = std.(eachcol(data_cols)),
        Q1 = map(col -> quantile(col, 0.25), eachcol(data_cols)),
        Q3 = map(col -> quantile(col, 0.75), eachcol(data_cols)),
        Min = minimum.(eachcol(data_cols)),
        Max = maximum.(eachcol(data_cols))
    )
    CSV.write("$(path)/$(eval_kind)_stats.csv", df_results)
    println("$(path)/$(eval_kind)_stats.csv:\n$df_results\n")
end
