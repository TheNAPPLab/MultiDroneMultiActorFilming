using SubmodularMaximization
using MDMA
using DataFrames
using Images
using FileIO
using Debugger
using CSV

export evaluate_solution, evaluate_ptz_solution, PPAEvaluation, ImageEvaluation, SceneCoverageEvaluation, ActorCoverageEvaluation, ActorRewardEvaluation

struct ImageEvaluation end
struct PPAEvaluation end
struct SceneCoverageEvaluation end
struct ActorCoverageEvaluation end
struct ActorRewardEvaluation end

# Computes view quality for images as surface integral of square root pixel
# density with unweighted faces
#
# sum over faces of
#   face_area * sqrt(pixels / face_area)
function cumulative_view_quality(counts_by_color)
    reward = 0

    side_area = 1.01 # m^2
    top_area = 0.811 # m^2
    # Compute PPA for this timestep based on face pixel counts
    for (pixel, count) in counts_by_color
        # We are looking at a top face if the red channel is > 0.5
        # In that case use the known face area for the top face
        # else use the face area for the side
        # previous_count = i == 1 ? 0 : seen_faces[pixel]
        view_quality = x -> red(pixel) > 0.5 ? top_area * sqrt(x / top_area) : side_area * sqrt(x / side_area)

        # Could add a multiplicative factor on count to handle resolution (micah)
        reward += view_quality(count)
    end
    return reward
end

function evaluate_solution(
    experiment_name::String,
    planner_types::Vector{String},
    path_to_experiments::String,
    _::ImageEvaluation,
)::DataFrame

    root_path = "$(path_to_experiments)/$(experiment_name)"
    blender_path = "$(root_path)/blender_output"

    multi_configs =
        configs_from_file("$(root_path)/$(experiment_name)_data.json", experiment_name, 3.0)

    horizon = multi_configs.horizon


    df = DataFrame(t = 1:horizon)
    for planner in planner_types
        cameras = readdir("$(blender_path)/$(planner)", join = true)
        evals = []
        for t = 1:horizon
            images_at_t = []
            for camera in cameras
                image_paths = readdir(camera, join = true)
                push!(images_at_t, image_paths[t])
            end
            println(images_at_t)
            # # Value for this timestep in the plot
            # val_now = 0.0
            # Contains pixels -> num
            accumulated_pixels = Dict()
            for (_,img_path) in enumerate(images_at_t)
                # For every image
                img = load(img_path)

                # Collect the unique colors in the image
                for pixel in img
                    mag = (red(pixel) + green(pixel) + blue(pixel))/3.
                    # Skip very dark or bright pixels
                    if (red(pixel) ==  blue(pixel) == green(pixel))
                        continue
                    end

                    if !haskey(accumulated_pixels, pixel)
                        push!(accumulated_pixels, pixel => 1)
                    else
                        accumulated_pixels[pixel] += 1
                    end
                end
            end

            reward = cumulative_view_quality(accumulated_pixels)
            push!(evals, reward)
        end

        df[!, planner] = evals
    end

    df
end

function evaluate_solution(
    experiment_name::String,
    planner_types::Vector{String},
    path_to_experiments::String,
    _::PPAEvaluation,
)::DataFrame

    root_path = "$(path_to_experiments)/$(experiment_name)"

    # Just using this to get the trajectories and horizon
    multi_configs =
        configs_from_file("$(root_path)/$(experiment_name)_data.json", experiment_name, "$(root_path)/$(experiment_name)_ptz_data.json", 1000.0)

    target_trajectories = multi_configs.target_trajectories
    horizon = multi_configs.horizon
    sensor = MDMA.PinholeCameraModel(4.4, [1920, 1080], [5.60, 3.15], 1000.0)

    # Make dataframe
    df = DataFrame(t = 1:horizon)
    for planner in planner_types
        solution = load_solution("$(root_path)/$(planner)/solution.json")

        # Get evaluation for all
        view_quality_values = []

        # Might need to change the evaluation method.
        # First robot coverage array is zero
        # We have its trajectory so we can compute coverage data for the second robot
        # Lets compute the coverage on a per timestep basis if possible
        # In aggregate
        coverage_data = generate_empty_coverage_data(multi_configs)
        for t = 1:horizon
            view_quality_at_time = 0.0
            states = []

            # Would be nice to do this in a neater way
            # Get 3 states, one for each robot at this point in time
            for robot in solution.elements
                push!(states, robot[2][t])
            end

            targets = target_trajectories[t, :]
            for (target_id, target) in enumerate(targets)
                for (f_id, face) in enumerate(target.faces)
                    for state in states
                        if detectTarget(state.state, target, sensor)
                            distance = (
                                face.pos[1] - state.state.x,
                                face.pos[2] - state.state.y,
                                -target_height / 2 + state.state.z,
                            )
                            theta = (2 * pi) - state.state.pan
                            look_direction = (cos(theta), sin(theta), 0.0)
                            # Set previous_coverage to zero
                            camera_pixel_density = compute_camera_coverage(
                                face,
                                look_direction,
                                distance
                            )
                            coverage_data[t, target_id, f_id] += camera_pixel_density
                        end
                    end
                    # Accumulate reward for the face after we finish
                    view_quality =
                        face_view_quality(face,
                                          coverage_data[t, target_id, f_id])
                    view_quality_at_time += view_quality
                end
            end
            push!(view_quality_values, view_quality_at_time)
        end

        df[!, planner] = view_quality_values
    end
    return df
end

function evaluate_ptz_solution(
    experiment_name::String,
    solution_filenames::Vector{String},
    solution_labels::Vector{String},
    path_to_experiments::String,
    study::String,
    _::PPAEvaluation,
)::DataFrame

    root_path = "$(path_to_experiments)/$(experiment_name)"

    # Just using this to get the trajectories and horizon
    multi_configs =
        configs_from_file("$(root_path)/$(experiment_name)_data.json", experiment_name, "$(root_path)/$(experiment_name)_ptz_data_$(study).json", 1000.0)

    target_trajectories = multi_configs.target_trajectories
    horizon = multi_configs.horizon
    sensors = multi_configs.grid.pinhole_cameras

    # Make dataframe
    df = DataFrame(t = 1:horizon)
    for (i, filename) in enumerate(solution_filenames)

        println("\n\nEvaluating solution: $filename\n\n")

        solution = load_solution("$(root_path)/greedy/$(filename).json")

        # Get evaluation for all
        view_quality_values = []

        # Might need to change the evaluation method.
        # First robot coverage array is zero
        # We have its trajectory so we can compute coverage data for the second robot
        # Lets compute the coverage on a per timestep basis if possible
        # In aggregate
        coverage_data = generate_empty_coverage_data(multi_configs)
        for t = 1:horizon
            view_quality_at_time = 0.0
            states = []

            # Would be nice to do this in a neater way
            # Get 3 states, one for each robot at this point in time
            for robot in solution.elements
                push!(states, robot[2][t])
            end

            targets = target_trajectories[t, :]
            for (target_id, target) in enumerate(targets)
                for (f_id, face) in enumerate(target.faces)
                    for state in states
                        index = findfirst(x -> x == (state.state.x, state.state.y, state.state.z), multi_configs.grid.camera_positions)
                        if detectTarget(state.state, target, sensors[index])
                            distance = (
                                face.pos[1] - state.state.x,
                                face.pos[2] - state.state.y,
                                -target_height / 2 + state.state.z,
                            )
                            pan = -state.state.pan - multi_configs.grid.pinhole_cameras[index].pan_offset
                            tilt = state.state.tilt + multi_configs.grid.pinhole_cameras[index].tilt_offset
                            zoom = state.state.zoom

                            look_direction = (cos(pan)*cos(tilt), sin(pan)*cos(tilt), -sin(tilt))

                            # Set previous_coverage to zero
                            alpha = compute_alpha(multi_configs.grid.pinhole_cameras[index], zoom)
                            camera_pixel_density = compute_camera_coverage(
                                face,
                                look_direction,
                                distance,
                                alpha
                            )
                            coverage_data[t, target_id, f_id] += camera_pixel_density
                        end
                    end
                    # Accumulate reward for the face after we finish
                    view_quality =
                        face_view_quality(face,
                                          coverage_data[t, target_id, f_id])
                    view_quality_at_time += view_quality
                end
            end
            push!(view_quality_values, view_quality_at_time)
        end

        df[!, solution_labels[i]] = view_quality_values
    end
    return df
end

function evaluate_ptz_solution(
    experiment_name::String,
    solution_filenames::Vector{String},
    solution_labels::Vector{String},
    path_to_experiments::String,
    study::String,
    _::SceneCoverageEvaluation,
)::DataFrame

    root_path = "$(path_to_experiments)/$(experiment_name)"

    # Just using this to get the trajectories and horizon
    multi_configs =
        configs_from_file("$(root_path)/$(experiment_name)_data.json", experiment_name, "$(root_path)/$(experiment_name)_ptz_data_$(study).json", 1000.0)

    target_trajectories = multi_configs.target_trajectories
    horizon = multi_configs.horizon
    sensors = multi_configs.grid.pinhole_cameras

    # Make dataframe
    df = DataFrame(t = 1:horizon)
    for (i, filename) in enumerate(solution_filenames)

        println("\n\nEvaluating solution: $filename\n\n")

        solution = load_solution("$(root_path)/greedy/$(filename).json")

        # Number of actors in view of at least one camera at each time step
        covered_actors = []
        for t = 1:horizon
            num_targets_at_time = 0
            states = []

            # Get camera states at this time step
            for robot in solution.elements
                push!(states, robot[2][t])
            end

            targets = target_trajectories[t, :]
            for (target_id, target) in enumerate(targets)
                target_covered = false
                for state in states
                    index = findfirst(x -> x == (state.state.x, state.state.y, state.state.z), multi_configs.grid.camera_positions)
                    if detectTarget(state.state, target, sensors[index]) && !target_covered
                        num_targets_at_time += 1
                        target_covered = true
                    end
                end
            end
            push!(covered_actors, num_targets_at_time)
        end
        
        df[!, solution_labels[i]] = covered_actors
    end
    return df
end

function evaluate_ptz_solution(
    experiment_name::String,
    solution_filename::String,
    path_to_experiments::String,
    study::String,
    _::ActorCoverageEvaluation,
)::DataFrame

    root_path = "$(path_to_experiments)/$(experiment_name)"

    # Get the trajectories, horizon, and sensors
    multi_configs =
        configs_from_file("$(root_path)/$(experiment_name)_data.json", experiment_name, "$(root_path)/$(experiment_name)_ptz_data_$(study).json", 1000.0)
    target_trajectories = multi_configs.target_trajectories
    horizon = multi_configs.horizon
    sensors = multi_configs.grid.pinhole_cameras

    # Make dataframe
    df = DataFrame(t = 1:horizon)

    println("\n\nEvaluating solution: $solution_filename\n\n")

    solution = load_solution("$(root_path)/greedy/$(solution_filename).json")

    for target_id in axes(target_trajectories, 2)
        # Number of cameras viewing this target at each time step
        covering_cameras = []

        for t = 1:horizon
            num_cameras_at_time = 0
            states = []

            # Get camera states at this time step
            for robot in solution.elements
                push!(states, robot[2][t])
            end

            target = target_trajectories[t, target_id]
            for state in states
                index = findfirst(x -> x == (state.state.x, state.state.y, state.state.z), multi_configs.grid.camera_positions)
                if detectTarget(state.state, target, sensors[index])
                    num_cameras_at_time += 1
                end
            end
            push!(covering_cameras, num_cameras_at_time)
        end

        df[!, "Target $target_id"] = covering_cameras
    end
    return df
end

function evaluate_ptz_solution(
    experiment_name::String,
    solution_filename::String,
    path_to_experiments::String,
    study::String,
    _::ActorRewardEvaluation,
)::DataFrame

    root_path = "$(path_to_experiments)/$(experiment_name)"

    # Get the trajectories, horizon, and sensors
    multi_configs =
        configs_from_file("$(root_path)/$(experiment_name)_data.json", experiment_name, "$(root_path)/$(experiment_name)_ptz_data_$(study).json", 1000.0)
    target_trajectories = multi_configs.target_trajectories
    horizon = multi_configs.horizon
    sensors = multi_configs.grid.pinhole_cameras

    # Make dataframe
    df = DataFrame(t = 1:horizon)

    println("\n\nEvaluating solution: $solution_filename\n\n")

    solution = load_solution("$(root_path)/greedy/$(solution_filename).json")

    # Determine the objective reward for each target at each time step
    for target_id in axes(target_trajectories, 2)
        # Evaluation for the actor at every time step
        view_quality_values = []

        coverage_data = generate_empty_coverage_data(multi_configs)
        for t = 1:horizon
            view_quality_at_time = 0.0
            states = []

            # Get camera states at this time step
            for robot in solution.elements
                push!(states, robot[2][t])
            end
            
            target = target_trajectories[t, target_id]
            for (f_id, face) in enumerate(target.faces)
                for state in states
                    index = findfirst(x -> x == (state.state.x, state.state.y, state.state.z), multi_configs.grid.camera_positions)
                    if detectTarget(state.state, target, sensors[index])
                        distance = (
                            face.pos[1] - state.state.x,
                            face.pos[2] - state.state.y,
                            -target_height / 2 + state.state.z,
                        )
                        pan = -state.state.pan - sensors[index].pan_offset
                        tilt = state.state.tilt + sensors[index].tilt_offset
                        zoom = state.state.zoom

                        look_direction = (cos(pan)*cos(tilt), sin(pan)*cos(tilt), -sin(tilt))

                        # Set previous_coverage to zero
                        alpha = compute_alpha(sensors[index], zoom)
                        camera_pixel_density = compute_camera_coverage(
                            face,
                            look_direction,
                            distance,
                            alpha
                        )
                        coverage_data[t, target_id, f_id] += camera_pixel_density
                    end
                end
                # Accumulate reward for the face after we finish
                view_quality =
                    face_view_quality(face,
                                        coverage_data[t, target_id, f_id])
                view_quality_at_time += view_quality
            end
            push!(view_quality_values, view_quality_at_time)
        end

        df[!, "Target $target_id"] = view_quality_values
    end
    return df
end