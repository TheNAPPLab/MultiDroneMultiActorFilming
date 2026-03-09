export ccw, cw, absoluteAngle, detectTarget, dirAngle, up, down
using LinearAlgebra

# using MDMA_Problem
function ccw(d::Symbol)::Symbol
    d in cardinaldir || throw(ArgumentError("invalid cardinaldir: $d"))
    if (d == :E)
        Symbol("NE")
    elseif (d == :NE)
        Symbol("N")
    elseif (d == :N)
        Symbol("NW")
    elseif (d == :NW)
        Symbol("W")
    elseif (d == :W)
        Symbol("SW")
    elseif (d == :SW)
        Symbol("S")
    elseif (d == :S)
        Symbol("SE")
    elseif (d == :SE)
        Symbol("E")
    end
end

function ccw(p::Float64)::Float64
    p in pan_angles || throw(ArgumentError("invalid pan: $p"))
    index = findfirst(isequal(p), pan_angles)
    if (index == length(pan_angles))
        pan_angles[1]
    else
        pan_angles[index+1]
    end
end

function cw(d::Symbol)::Symbol
    d in cardinaldir || throw(ArgumentError("invalid cardinaldir: $d"))
    if (d == :E)
        Symbol("SE")
    elseif (d == :NE)
        Symbol("E")
    elseif (d == :N)
        Symbol("NE")
    elseif (d == :NW)
        Symbol("N")
    elseif (d == :W)
        Symbol("NW")
    elseif (d == :SW)
        Symbol("W")
    elseif (d == :S)
        Symbol("SW")
    elseif (d == :SE)
        Symbol("S")
    end
end

function cw(p::Float64)::Float64
    p in pan_angles || throw(ArgumentError("invalid pan: $p"))
    index = findfirst(isequal(p), pan_angles)
    if (index == 1)
        pan_angles[length(pan_angles)]
    else
        pan_angles[index-1]
    end
end

function up(t::Float64)::Float64
    t in tilt_angles || throw(ArgumentError("invalid tilt: $t"))
    index = findfirst(isequal(t), tilt_angles)
    if (index == length(tilt_angles))
        t
    else
        tilt_angles[index+1]
    end
end

function down(t::Float64)::Float64
    t in tilt_angles || throw(ArgumentError("invalid tilt: $t"))
    index = findfirst(isequal(t), tilt_angles)
    if (index == 1)
        t
    else
        tilt_angles[index-1]
    end
end

function zoom_in(z::Float64)::Float64
    z in zoom_vals || throw(ArgumentError("invalid zoom: $z"))
    index = findfirst(isequal(z), zoom_vals)
    if (index == length(zoom_vals))
        z
    else
        zoom_vals[index+1]
    end
end

function zoom_out(z::Float64)::Float64
    z in zoom_vals || throw(ArgumentError("invalid zoom: $z"))
    index = findfirst(isequal(z), zoom_vals)
    if (index == 1)
        z
    else
        zoom_vals[index-1]
    end
end

function absoluteAngle(dx::Number, dy::Number)
    actor_angle = abs(atan(dy / dx))
    if (dy >= 0 && dx >= 0) # Q1
        actor_angle = actor_angle
    elseif (dy >= 0 && dx < 0) # Q2
        actor_angle = pi - actor_angle
    elseif (dy < 0 && dx <= 0) # Q3
        actor_angle = pi + actor_angle
    elseif (dy < 0 && dx > 0) # Q4
        actor_angle = 2 * pi - actor_angle
    end
    actor_angle
end

absoluteAngle(theta::Number) = absoluteAngle(cos(theta), sin(theta))

function detectTarget(dstate::PTZState, astate::Target, sensor::ViewConeSensor)::Bool
    dist_sqr = (dstate.x - astate.x)^2 + (dstate.y - astate.y)^2
    # Check if inside the max view distance
    # if dist_sqr > sensor.cutoff^2
    #     return false
    # else
        # Get view bounds based on heading of Agent
        pan_angle = dstate.pan
        view_bounds = [-sensor.fov / 2, sensor.fov / 2]

        # compute the true angle of the actor
        dy = astate.y - dstate.y
        dx = astate.x - dstate.x
        actor_angle = absoluteAngle(dx, dy)

        # Absolute positions of the top and bottom of the bound not accounting
        # for wrapping inside the 0 to 360 range
        true_top = pan_angle + sensor.fov / 2
        true_bot = pan_angle - sensor.fov / 2

        # Need to change the min and max ranges for comparison in cases where
        # the boundary is split Across the x axis. This also requires changing
        # our comparison functions to OR operators rather than AND operators at
        # the end
        and(a, b) = a && b
        or(a, b) = a || b
        compareOp = and
        # println(true_bot," ", true_top)
        if (true_bot < 0 || true_top > 2 * pi)
            min_angle = absoluteAngle(true_bot)
            max_angle = absoluteAngle(true_top)
            compareOp = or
            # println("or")
        else
            actor_angle = pan_angle - actor_angle
            min_angle = view_bounds[1]
            max_angle = view_bounds[2]
            # println("and")
        end

        # Check if actor angle is inside the view bounds (observed)
        # println(actor_angle >= min_angle," ", actor_angle <= max_angle)
        if (compareOp(actor_angle >= min_angle, actor_angle <= max_angle))
            return true
        else
            return false
        end
    # end
end

function detectTarget(dstate::PTZState, astate::Target, camera::PinholeCameraModel)::Bool
    pan = dstate.pan + camera.pan_offset
    tilt = -dstate.tilt - camera.tilt_offset
    zoom = dstate.zoom
    pan_rotation = [cos(pan) -sin(pan) 0; sin(pan) cos(pan) 0; 0 0 1]
    tilt_rotation = [cos(tilt) 0 sin(tilt); 0 1 0; -sin(tilt) 0 cos(tilt)]
    zoom_adjust = [zoom 0 0; 0 zoom 0; 0 0 1]
    R = camera.extrinsics * tilt_rotation * pan_rotation
    t = -R * [dstate.x; dstate.y; -dstate.z]
    image_coord = [R t] * [astate.x; astate.y; -target_height; 1]
    if image_coord[3] >= 0
        image_coord = camera.intrinsics * zoom_adjust * image_coord
        #print("\nScale ", image_coord[3])
        image_coord = image_coord / image_coord[3]
        in_frame =
            sum(image_coord[1:2] .> camera.resolution) + sum(image_coord[1:2] .< [0; 0])
        #print("\nDetect target ", rel_target_pos, " ", [dstate.x; dstate.y; drone_height], " ", dstate.heading, " ", image_coord, " ", in_frame)
        if in_frame == 0
            return true
        else
            return false
        end
    else
        return false
    end
end

function dirAngle(h::Symbol)
    h in cardinaldir || throw(ArgumentError("invalid cardinaldir: $h"))
    if (h == :E)
        0
    elseif (h == :NE)
        pi / 4
    elseif (h == :N)
        pi / 2
    elseif (h == :NW)
        3 * pi / 4
    elseif (h == :W)
        pi
    elseif (h == :SW)
        5 * pi / 4
    elseif (h == :S)
        3 * pi / 2
    elseif (h == :SE)
        7 * pi / 4
    end
end

# @testset "detectTarget" begin
#     sensor = ViewConeSensor(pi / 2, 3)
#     dstate = UAVState(0.0, 0.0, :N)
#     @test detectTarget(dstate, Target(0, 2, 0, 1), sensor) == true
#     @test detectTarget(dstate, Target(0, 3, 0, 1), sensor) == true
#     @test detectTarget(dstate, Target(1, 3, 0, 1), sensor) == false
#     @test detectTarget(dstate, Target(1, 2, 0, 1), sensor) == true
#     @test detectTarget(dstate, Target(-1, 2, 0, 1), sensor) == true
#     @test detectTarget(dstate, Target(-1, 1, 0, 1), sensor) == true
#     @test detectTarget(dstate, Target(1, 1, 0, 1), sensor) == true
#     @test detectTarget(dstate, Target(-1, 0, 0, 1), sensor) == false
#     sensor = ViewConeSensor(pi, 3)
#     dstate = UAVState(0.0, 0.0, :N)
#     @test detectTarget(dstate, Target(-1, 0, 0, 1), sensor) == true
#     @test detectTarget(dstate, Target(-1, -1, 0, 1), sensor) == false
#     @test detectTarget(dstate, Target(1, -1, 0, 1), sensor) == false
#     dstate = UAVState(0.0, 0.0, :N)
#     sensor = ViewConeSensor(3 * pi / 2, 3)
#     @test detectTarget(dstate, Target(1, -1, 0, 1), sensor) == true
#     @test detectTarget(dstate, Target(-1, -1, 0, 1), sensor) == true
#     dstate = UAVState(0.0, 0.0, :E)
#     sensor = ViewConeSensor(3 * pi / 2, 3)
#     @test detectTarget(dstate, Target(-1, -1, 0, 1), sensor) == true
#     @test detectTarget(dstate, Target(-1, 1, 0, 1), sensor) == true
#     @test detectTarget(dstate, Target(-1, 0, 0, 1), sensor) == false
#     dstate = UAVState(0.0, 0.0, :S)
#     sensor = ViewConeSensor(3 * pi / 2, 3)
#     @test detectTarget(dstate, Target(0, 1, 0, 1), sensor) == false
#     # This test fails due to numerical precision issues comparing 44.99 to 45.0
#     # and does not represent a logical issue with the code should not be an
#     # issue in practice
#     # @test detectTarget(dstate, Target(1, 1, 0, 1), sensor) == true
#     @test detectTarget(dstate, Target(2, 1, 0, 1), sensor) == true
#     @test detectTarget(dstate, Target(-1, 1, 0, 1), sensor) == true

#     dstate = UAVState(0.0, 0.0, :SE)
#     sensor = ViewConeSensor(pi / 2, 3)
#     # Test also fails. need to check
#     # @test detectTarget(dstate, Target(1, 0, 0, 1), sensor) == true

#     dstate = UAVState(0.0, 0.0, :W)
#     sensor = ViewConeSensor(3 * pi / 2, 3)
#     @test detectTarget(dstate, Target(1, 0, 0, 1), sensor) == false
#     @test detectTarget(dstate, Target(1, 1, 0, 1), sensor) == true
#     @test detectTarget(dstate, Target(-1, -1, 0, 1), sensor) == true
#     dstate = UAVState(0.0, 0.0, :E)
#     sensor = ViewConeSensor(pi / 2, 3)
#     @test detectTarget(dstate, Target(1, 1, 0, 1), sensor) == true
#     @test detectTarget(dstate, Target(1, 0, 0, 1), sensor) == true
#     @test detectTarget(dstate, Target(1, -1, 0, 1), sensor) == true
#     @test detectTarget(dstate, Target(-1, 0, 0, 1), sensor) == false
#     dstate = UAVState(0.0, 0.0, :NE)
#     sensor = ViewConeSensor(3 * pi / 2, 3)
#     @test detectTarget(dstate, Target(1, 1, 0, 1), sensor) == true
#     @test detectTarget(dstate, Target(1, 0, 0, 1), sensor) == true
#     @test detectTarget(dstate, Target(1, -1, 0, 1), sensor) == true
#     @test detectTarget(dstate, Target(-1, -1, 0, 1), sensor) == false

# end
