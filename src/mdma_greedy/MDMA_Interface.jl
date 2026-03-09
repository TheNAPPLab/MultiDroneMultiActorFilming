# This file contains code for interfacing with Airsim

import JSON3
using Test
using SubmodularMaximization
using MDMA

export configs_from_file, save_solution, load_solution, targets_from_file


function configs_from_file(
    filename::String,
    experiment_name::String,
    ptz_data::String,
    cutoff::Number
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
    sense_dist = Float64(cutoff)

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

    # Get data for PTZ cameras
    json_string = read(ptz_data, String)
    json_root = JSON3.read(json_string)

    camera_data = json_root["camera_data"]
    positions = camera_data["positions"]
    pt_offset = camera_data["pt_offset"]
    pan_divisions = camera_data["pan_divisions"]
    tilt_divisions = camera_data["tilt_divisions"]
    zoom_divisions = camera_data["zoom_divisions"]
    pinhole_params = json_root["pinhole_params"]
    focal_length = pinhole_params["focal_length"]
    resolution = pinhole_params["resolution"]
    lens_dim = pinhole_params["lens_dim"]
    ptz_limits = json_root["ptz_limits"]
    pan = ptz_limits["pan"]
    tilt = ptz_limits["tilt"]
    zoom = ptz_limits["zoom"]

    # Get positions and pinhole camera models for each camera
    camera_positions = Tuple{Float64,Float64,Float64}[]
    pinhole_cameras = PinholeCameraModel[]
    for i in eachindex(positions)
        push!(camera_positions, (positions[i][1], positions[i][2], positions[i][3]))
        new_sensor = PinholeCameraModel(focal_length, [resolution[1], resolution[2]], [lens_dim[1], lens_dim[2]], sense_dist, Float64(pt_offset[i][1]), Float64(pt_offset[i][2]))
        push!(pinhole_cameras, new_sensor)
    end

    # Discretize pan, tilt, zoom
    discretize_pan(pan_divisions, pan[1], pan[2])
    discretize_tilt(tilt_divisions, tilt[1], tilt[2])
    discretize_zoom(zoom_divisions, zoom[1], zoom[2])

    # Making the object
    grid = MDMA_Grid(Int64(scale["x"]), Int64(scale["y"]), camera_positions, pinhole_cameras, pan_divisions, tilt_divisions, zoom_divisions, horizon)
    sensor = PinholeCameraModel(focal_length, [resolution[1], resolution[2]], [lens_dim[1], lens_dim[2]], Float64(sense_dist))

    return MultiDroneMultiActorConfigs(
        experiment_name = experiment_name,
        num_robots = num_robots,
        target_trajectories = target_trajectories,
        grid = grid,
        sensor = sensor,
        horizon = horizon,
        move_dist = 0.0,
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
