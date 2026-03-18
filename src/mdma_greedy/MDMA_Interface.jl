# This file contains code for interfacing with Airsim

import JSON3
using Test
using SubmodularMaximization
using MDMA

export configs_from_file, save_solution, load_solution, targets_from_file


# Convenience wrapper: auto-generates camera positions from scene bounds and
# uses default pan/tilt/zoom discretizations.
function configs_from_file(
    filename::String,
    experiment_name::String,
    move_dist::Number,
)::MultiDroneMultiActorConfigs

    json_string = read(filename, String)
    json_root   = JSON3.read(json_string)

    num_robots = Int(json_root["num_robots"])
    scale      = json_root["scene"]["scale"]
    w          = Float64(scale["x"])
    h          = Float64(scale["y"])

    # Estimate coordinate bounds from the first frame of actor positions
    first_frame = json_root["actor_positions"][1]
    xs = [Float64(pos["location"][1]) for pos in first_frame]
    ys = [Float64(pos["location"][2]) for pos in first_frame]
    x_min, x_max = minimum(xs), maximum(xs)
    y_min, y_max = minimum(ys), maximum(ys)

    # Add padding so cameras aren't right at the edge
    x_pad = max((x_max - x_min) * 0.15, 1.0)
    y_pad = max((y_max - y_min) * 0.15, 1.0)
    x_min -= x_pad;  x_max += x_pad
    y_min -= y_pad;  y_max += y_pad

    # Camera height scaled relative to scene size
    camera_height = max(w, h) / 6.0

    # Distribute cameras in a roughly square grid across the scene
    n     = max(num_robots, 1)
    ncols = ceil(Int, sqrt(Float64(n)))
    nrows = ceil(Int, n / ncols)

    camera_positions = Tuple{Float64,Float64,Float64}[]
    count = 0
    for row in 1:nrows
        for col in 1:ncols
            if count < n
                x = x_min + (col - 0.5) * (x_max - x_min) / ncols
                y = y_min + (row - 0.5) * (y_max - y_min) / nrows
                push!(camera_positions, (x, y, camera_height))
                count += 1
            end
        end
    end

    pan_divisions  = 8
    tilt_divisions = 8
    zoom_divisions = 3

    configs_from_file(
        filename, experiment_name, move_dist,
        camera_positions, pan_divisions, tilt_divisions, zoom_divisions,
    )
end


function configs_from_file(
    filename::String,
    experiment_name::String,
    move_dist::Number,
    camera_positions::Vector{Tuple{Float64,Float64,Float64}},
    pan_divisions::Int64,
    tilt_divisions::Int64,
    zoom_divisions::Int64
)::MultiDroneMultiActorConfigs

    json_string = read(filename, String)
    json_root = JSON3.read(json_string)

    scene = json_root["scene"]
    scale = scene["scale"]
    blend_file = json_root["filename"]
    horizon = json_root["num_frames"]
    num_targets = json_root["num_targets"]
    robot_fovs = json_root["robot_fovs"]
    num_robots = json_root["num_robots"]
    sense_dist = json_root["sense_dist"]

    min_x = Inf
    max_x = -Inf
    min_y = Inf
    max_y = -Inf
    for target_set in json_root["actor_positions"]
        for loc_pos in target_set
            loc = loc_pos["location"]
            x = Float64(loc[1])
            y = Float64(loc[2])
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
        end
    end

    grid_width = Int64(scale["x"])
    grid_height = Int64(scale["y"])
    shift_x = 0.0
    shift_y = 0.0

    needs_normalization = (min_x < 1.0) || (min_y < 1.0) || (max_x > grid_width) || (max_y > grid_height)
    if needs_normalization
        pad = 2.0
        shift_x = -min_x + 1.0 + pad
        shift_y = -min_y + 1.0 + pad
        grid_width = Int64(ceil((max_x - min_x) + (2 * pad) + 2.0))
        grid_height = Int64(ceil((max_y - min_y) + (2 * pad) + 2.0))
        camera_positions = map(p -> (p[1] + shift_x, p[2] + shift_y, p[3]), camera_positions)
    end

    target_trajectories = Array{Target,2}(undef, horizon, num_targets)
    for (time, target_set) in enumerate(json_root["actor_positions"])
        target_row = Vector{Target}(undef, num_targets)
        for (id, loc_pos) in enumerate(target_set)
            loc = loc_pos["location"]
            rot = loc_pos["rotation"]
            x = Float64(loc[1]) + shift_x
            y = Float64(loc[2]) + shift_y
            h = Float64(rot[3]) # Take rotation around z as the "pan"
            weight = Float64(loc_pos["weight"])
            target_row[id] = multiply_face_weights(Target(x, y, h, id), weight)
        end
        target_trajectories[time, :] = target_row
    end


    # Making the object
    grid = MDMA_Grid(grid_width, grid_height, camera_positions, pan_divisions, tilt_divisions, zoom_divisions, horizon)
    fov = robot_fovs[1]
    sensor = PinholeCameraModel(4.4, [1920.0, 1080.0], [5.60, 3.15], 0.0, Float64(sense_dist))

    return MultiDroneMultiActorConfigs(
        experiment_name = experiment_name,
        num_robots = num_robots,
        target_trajectories = target_trajectories,
        grid = grid,
        sensor = sensor,
        horizon = horizon,
        move_dist = move_dist,
    )

end

function targets_from_file(filename::String)::Array{Target,2}

    json_string = read(filename, String)
    json_root = JSON3.read(json_string)

    horizon = json_root["num_frames"]
    num_targets = json_root["num_targets"]

    target_trajectories = Array{Target,2}(undef, horizon, num_targets)
    for (time, target_set) in enumerate(json_root["actor_positions"])
        target_row = Vector{Target}(undef, num_targets)
        for (id, loc_pos) in enumerate(target_set)
            loc = loc_pos["location"]
            rot = loc_pos["rotation"]
            x = loc[1]
            y = loc[2]
            h = rot[3] # Take rotatin around z as the "pan"
            weight = loc_pos["weight"]
            target_row[id] = multiply_face_weights(Target(x, y, h, id), weight)
        end
        target_trajectories[time, :] = target_row
    end

    return target_trajectories
end

# Serialize the solution into a json object
function save_solution(
    experiment_name::String,
    path_to_experiments::String,
    subdir::String,
    solution::Solution,
    multi_configs::MultiDroneMultiActorConfigs,
)
    root_dict = Dict("value" => solution.value, "elements" => solution.elements)

    directory = "$(path_to_experiments)/$(experiment_name)/$(subdir)"
    mkpath(directory)
    mkpath("$(directory)/renders")
    open("$(directory)/solution.json", "w") do io
        JSON3.pretty(io, root_dict)
    end

    println("Saving solution to $(directory)/solution.json")
    render_paths(solution, multi_configs, "$(directory)/renders")

end


function load_solution(filename)
    json_string = read(filename, String)
    root_dict = JSON3.read(json_string)

    solution_value = root_dict["value"]
    solution_elements_dict = root_dict["elements"]
    # Solution Elements is an array of array of (Tuple, Vector{MDPState})

    elements = Tuple{Int64,Vector{MDPState}}[]
    for robot_dict in solution_elements_dict
        robot_id = robot_dict[1]
        robot_states = MDPState[]
        for states in robot_dict[2]
            state_dict = states["state"]
            depth = states["depth"]
            horizon = states["horizon"]
            state =
                PTZState(Float64(state_dict["x"]), Float64(state_dict["y"]), Float64(state_dict["z"]), Float64(state_dict["pan"]), Float64(state_dict["tilt"]), Float64(state_dict["zoom"]))
            push!(robot_states, MDPState(state, depth, horizon))
        end
        push!(elements, (robot_id, robot_states))
    end

    Solution(Float64(solution_value), elements)

end
