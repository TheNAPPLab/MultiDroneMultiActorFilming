# This file contains shared type aliases between Multi Agent and Single Agent Code

using Random

export CoverageData
export Trajectory, MDPState, State, Sensor, MDMA_Grid, random_state
# This is a 2D array since the rows represent timestamps For example this is
# what the data looks like if you have a 4 timestep scene in which 0 means not
# covered and 1 means covered
# Coverage Data
# | Time | Target1 | Target2 | Target3 |
# |    1 | (T1,0)  | (T2,1)  | (T3,1)  |
# |    2 | (T1,0)  | (T2,1)  | (T3,1)  |
# |    3 | (T1,1)  | (T2,1)  | (T3,1)  |
# |    4 | (T1,1)  | (T2,1)  | (T3,1)  |
const CoverageData = Array{Float64,3}

# Single Agent Types
const State = PTZState
const Sensor = PinholeCameraModel

struct MDPState
    state::State
    depth::Int64
    horizon::Int64
end

MDPState(state, horizon) = MDPState(state, 1, horizon)
MDPState(m::MDPState, s::State) = MDPState(s, m.depth + 1, m.horizon)
MDPState(m::MDPState) = MDPState(m.state, m.depth + 1, m.horizon)
MDPState(m::MDPState, a::MDPState) = MDPState(a.state, m.depth + 1, m.horizon)

const Trajectory = Vector{MDPState}

# State and trajectory objects for convenience
struct MDMA_Grid
    width::Int64
    height::Int64
    camera_positions::Vector{Tuple{Float64,Float64,Float64}}
    pan_divisions::Int64
    tilt_divisions::Int64
    zoom_divisions::Int64
    horizon::Int64
    states::Array{MDPState,5}

    # We will precompute some of the large objects that we use frequently
    function MDMA_Grid(width, height, camera_positions, pan_divisions, tilt_divisions, zoom_divisions, horizon)
        discretize_pan(pan_divisions)
        discretize_tilt(tilt_divisions)
        discretize_zoom(zoom_divisions)
        x = new(width, height, camera_positions, pan_divisions, tilt_divisions, zoom_divisions, horizon, get_states(camera_positions, pan_divisions, tilt_divisions, zoom_divisions, horizon))
        x
    end
end

function random_state(horizon, grid::MDMA_Grid)::MDPState
    rwidth = rand(1:grid.width)
    rheight = rand(1:grid.height)
    rdir = rand(cardinaldir)
    depth = 1

    MDPState(PTZState(rwidth, rheight, 0.0, rdir, 0.0 ,0.0), horizon)
end
