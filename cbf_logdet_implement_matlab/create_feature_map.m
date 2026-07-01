function map = create_feature_map(varargin)
%CREATE_FEATURE_MAP  Generate a set of 3D landmark positions in NED world frame.
%
%   map = create_feature_map()            uses all default parameters
%   map = create_feature_map(params)      struct with any subset of fields below
%
% =========================================================================
%  Parameters (fields of input struct, all optional)
% =========================================================================
%
%   n_features  [30]    Number of landmarks to generate
%   center      [3x1]   Centre of the feature volume in NED [m]  default [0;0;-3]
%   extent      [3x1]   Half-widths of the bounding box in NED [m]
%                       default [15; 15; 4]
%                       So features span:
%                           North : center(1) ± 15 m
%                           East  : center(2) ± 15 m
%                           Down  : center(3) ± 4  m
%   seed        [42]    Random seed for reproducibility (set [] for random)
%
% =========================================================================
%  NED Scene Layout
% =========================================================================
%
%   Drone typically flies near z = −3 m (3 m altitude in NED).
%   Features are placed in a flat-ish volume (smaller Down extent) to
%   mimic a room / outdoor environment with walls and ground features.
%
%               North
%                 ^
%                 |   [feature cloud]
%           ------+-------> East
%                 |
%              (drone trajectory in this plane)
%
% =========================================================================
%  Output struct fields
% =========================================================================
%
%   map.P_world   [3 x N]   Feature positions in NED world frame [m]
%   map.N         scalar    Total number of features
%   map.center    [3 x 1]   Centre used for generation
%   map.extent    [3 x 1]   Half-extents used for generation
%

%% --- Parse input ---
if nargin == 0
    params = struct();
else
    params = varargin{1};
end

N      = get_param(params, 'n_features', 30);
center = get_param(params, 'center',    [0; 0; -3]);   % NED [m]
extent = get_param(params, 'extent',    [15; 15; 4]);  % half-widths NED [m]
seed   = get_param(params, 'seed',      42);

%% --- Random generation ---
if ~isempty(seed)
    rng(seed);
end

% Uniform random in axis-aligned box
P_world = center + extent .* (2*rand(3, N) - 1);   % 3×N

%% --- Pack output ---
map.P_world = P_world;   % 3×N  landmark positions in NED
map.N       = N;
map.center  = center;
map.extent  = extent;

end % create_feature_map

% -------------------------------------------------------------------------
function val = get_param(s, field, default)
    if isfield(s, field)
        val = s.(field);
    else
        val = default;
    end
end
