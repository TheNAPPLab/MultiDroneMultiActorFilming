using Test
using LinearAlgebra
using SubmodularMaximization

export Camera,
    Target,
    ViewConeSensor,
    PinholeCameraModel,
    Face,
    rotMatrix,
    UAVState,
    PTZState,
    drone_height,
    target_height,
    multiply_face_weights

# Sensor representing a cone of vision from a drone
# Has an FOV as well as a maximum distance.
struct ViewConeSensor
    fov::Float64 # Radians representing FOV
    cutoff::Float64 # Max Distance
    println("ViewConeSensor Used")
end

# Sensor representing pinhole camera
struct PinholeCameraModel
    intrinsics::Matrix{Float64}
    extrinsics::Matrix{Float64}
    resolution::Vector{Int64}
    fov::Float64
    cutoff::Float64
    pan_offset::Float64 # In radians
    tilt_offset::Float64 # In radians
    function PinholeCameraModel(
        focal_length::Float64, # In mm
        resolution::Vector{Int64},
        lens_dim::Vector{Float64}, # In mm
        cutoff::Float64,
        pan::Float64 = 0.0,
        tilt::Float64 = 0.0
    )
        # world units to pixel units
        fx = focal_length * resolution[1] / lens_dim[1]
        fy = focal_length * resolution[2] / lens_dim[2]
        cx = resolution[1] / 2
        cy = resolution[2] / 2

        intrinsics = [fx 0 cx;
                      0 fy cy;
                      0 0 1]

        # One matrix is flipping from world coordinate to drone coordinate
        # then from drone to camera coordinate
        # Drone -> camera redifine the axis
        # Rotation on y axis
        extrinsics = [0 1 0; 0 0 1; 1 0 0]

        fov = 2 * atan(resolution[1], 2 * fx)
        println("PinholeCameraModel Used")
        println("1x Zoom Fov: $(fov)")
        return new(intrinsics, extrinsics, resolution, fov, cutoff, pan, tilt)
    end
end

Camera = Union{ViewConeSensor,PinholeCameraModel}

const drone_height::Float64 = 5.0 # meters
const target_height::Float64 = 1.8 # meters
const cardinaldir = Vector([:E, :NE, :N, :NW, :W, :SW, :S, :SE])

# Discretize pan, tilt, and zoom
pan_angles = Vector{Float64}(undef, 0) # In radians
tilt_angles = Vector{Float64}(undef, 0) # In radians
zoom_vals = Vector{Float64}(undef, 0) # x1, x2, etc

function discretize_pan(n::Int64, min_pan::Number, max_pan::Number)
    global pan_angles = Vector{Float64}(undef, n)
    # Convert degrees to radians
    min_pan = min_pan * pi / 180
    max_pan = max_pan * pi / 180
    increment = (max_pan - min_pan) / n
    for i = 0:(n-1)
        pan_angles[i+1] = min_pan + (i * increment)
    end
end

function discretize_tilt(n::Int64, min_tilt::Number, max_tilt::Number)
    global tilt_angles = Vector{Float64}(undef, n)
    # Convert degrees to radians
    min_tilt = min_tilt * pi / 180
    max_tilt = max_tilt * pi / 180
    increment = (max_tilt - min_tilt) / (n-1)
    for i = 0:(n-1)
        tilt_angles[i+1] = min_tilt + (i * increment)
    end
end

function discretize_zoom(n::Int64, min_zoom::Number, max_zoom::Number)
    global zoom_vals = Vector{Float64}(undef, n)
    increment = (max_zoom - min_zoom) / (n-1)
    for i = 0:(n-1)
        zoom_vals[i+1] = min_zoom + (i * increment)
    end
end

function dir_to_index(d::Symbol)
    if d in cardinaldir
        return findall(x -> x == d, cardinaldir)[1]
    end
end

function dir_to_index(d::Float64, options::Vector{Float64})
    if d in options
        return findall(x -> x == d, options)[1]
    end
end

# Target Faces
mutable struct Face
    normal::Vector{Float64} # 3d normal
    pos::Vector{Float64} # position
    size::Float64 # Size of face, total face area
    weight::Float64 # Observation Weight
    function Face(x::Float64, y::Float64, s::Float64, w::Float64, n::Vector{Float64})
        pos = [x; y]
        return new(n, pos, s, w)
    end
end


function rotMatrix(theta::Float64)
    [
        cos(theta) -sin(theta)
        sin(theta) cos(10)
    ]
end

# Target information
mutable struct Target
    x::Float64
    y::Float64
    heading::Float64
    apothem::Float64 # Distance from center to center of each face
    faces::Array{Face}
    nfaces::UInt32
    id::UInt32

    function Target(x::Number, y::Number, h::Number, a::Number, n::UInt32, id::UInt32)
        # n represents the number of total faces
        faces = Vector{Face}(undef, n)

        target_vertical_size = 1.8
        # Change in angle
        dphi = (2 * pi) / (n - 1)
        side_length = 2 * a * tan(dphi / 2)
        # Need to generate n faces with n normal vectors
        for i = 1:(n-1)
            # Account for heading, which is z rotation
            theta = (dphi * i) + h
            norm = [cos(theta); sin(theta); 0.0]
            pos = a * norm
            # Make sure faces are relative to Target position
            f = Face(x + pos[1], y + pos[2], side_length * target_vertical_size, 1.0, norm) # Possible weight 0.5*cos(theta+pi)+0.5
            # Set front faces to twice the weight
            # if i in 1:2 || i == n-1
            #     f.weight = 4.0
            # end
            faces[i] = f
        end

        # Adding top face
        norm = [0.0; 0.0; 1.0]
        # Top face should be in center of target
        f = Face(x, y, 3 * sqrt(3) * side_length^2 / 2, 1.0, norm)
        faces[n] = f


        new(Float64(x), Float64(y), h, a, faces, length(faces), id)
    end
end

function Target(x::Number, y::Number, h::Number, id::Number)
    Target(Float64(x), Float64(y), h, 0.968/2.0, UInt32(7), UInt32(id))
end

function multiply_face_weights(t::Target, weight::Number)::Target
    for f in t.faces
        f.weight *= weight
    end
    t
end
#  State struct for agents. Used specifically as part of the action space
struct UAVState
    x::Float64
    y::Float64
    heading::Symbol
    function UAVState(x::Float64, y::Float64, h::Symbol)
        h in cardinaldir || throw(ArgumentError("invalid cardinaldir: $h"))
        new(x, y, h)
    end
end

UAVState(x::Integer, y::Integer, h::Symbol) = UAVState(Float64(x), Float64(y), h)

# State struct for agents. Used specifically as part of the action space
struct PTZState
    x::Float64
    y::Float64
    z::Float64
    pan::Float64
    tilt::Float64
    zoom::Float64
    function PTZState(x::Float64, y::Float64, z::Float64, pan::Float64, tilt::Float64, zoom::Float64)
        pan in pan_angles || throw(ArgumentError("invalid pan: $pan"))
        tilt in tilt_angles || throw(ArgumentError("invalid tilt: $tilt"))
        zoom in zoom_vals || throw(ArgumentError("invalid zoom: $zoom"))
        new(x, y, z, pan, tilt, zoom)
    end
end

PTZState(x::Integer, y::Integer, z::Integer, pan::Float64, tilt::Integer, zoom::Integer) = PTZState(Float64(x), Float64(y), Float64(z), pan, Float64(tilt), Float64(zoom))

mutable struct ViewConeObservation
    n::Int64 # Number of actors detected
    distances::Vector{Float64} # Distances to actors
    faces::Vector{Face} # List of faces observed, not counting occlusions/etc
end


function drawTargets()
    f = Figure(resolution = (800, 800))
    Axis(f[1, 1], backgroundcolor = "black")

    xs = LinRange(-10, 10, 20)
    ys = LinRange(-10, 10, 20)
    t = Target(5.0, 5.0, 0.0, 5.0, UInt32(6), 1)

end
