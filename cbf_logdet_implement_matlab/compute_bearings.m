function [b_f, rho, eta, uvd, idx_vis, R_C_A] = compute_bearings(p_C_A, R_B_A, map_A, cam)
%COMPUTE_BEARINGS  Bearing vectors in anchor frame via camera projection.
%
%   [b_f, rho, eta, uvd, idx_vis, R_C_A] = compute_bearings(p_C_A, R_B_A, map_A, cam)
%
% =========================================================================
%  Frame Convention  (everything in anchor frame A)
% =========================================================================
%
%   Anchor frame A  :  a camera pose from the sliding window
%                      z = optical axis at anchor time
%   Body  frame B   :  FRD, rigidly attached to the drone
%   Camera frame C  :  OpenCV convention (x→right, y→down, z→forward)
%                      rigidly attached to body via cam.R_c2b
%
%   Chain:  camera ─R_c2b──▶ body ─R_B_A──▶ anchor
%           ⇒ camera ─R_C_A──▶ anchor   with R_C_A = R_B_A * R_c2b
%
%   For feature at position p_f^A in anchor frame:
%
%       r_A  = p_f^A − p_C^A            range vector in anchor frame
%       r_C  = R_C_A' * r_A             range vector in camera frame
%       ρ_j  = r_A(3)                   depth along anchor z-axis
%       b_{f,j} = r_A / ρ_j             bearing in anchor frame (b_{f,j}(3) = 1)
%       η_j  = ||b_{f,j}||              ≠ 1 in general
%
%   See cbf_logdet_triangulation.tex  Eq. (rjb_factored):
%       p_f^{C_j→A} = ρ · b_{f,j}
%
% =========================================================================
%  Inputs
% =========================================================================
%
%   p_C_A   [3×1]       Camera position in anchor frame
%   R_B_A   [3×3]       Body → anchor rotation matrix
%   map_A   struct      .P [3×N]  feature positions in anchor frame
%                        .N        number of features
%   cam     struct      Camera intrinsics:
%                        .fx, .fy, .cx, .cy, .W, .H, .R_c2b
%                        .rho_min, .rho_max, .sigma_bearing
%
% =========================================================================
%  Outputs
% =========================================================================
%
%   b_f      [3 × n_vis]  Bearing b_{f,j} in anchor frame (||·|| = η_j)
%   rho      [1 × n_vis]  Depth ρ_j = r_A(3), anchor-frame z-component
%   eta      [1 × n_vis]  η_j = ||b_{f,j}||
%   uvd      [3 × n_vis]  [u; v; Z] pixel projection + depth
%   idx_vis  [1 × n_vis]  Indices of visible features
%   R_C_A    [3 × 3]      Camera → anchor rotation (= R_B_A * R_c2b)
%

%% ── Camera-to-anchor rotation ───────────────────────────────────────────
R_c2b = get_cam(cam, 'R_c2b', [0,-1,0; 1,0,0; 0,0,1]);
R_C_A = R_B_A * R_c2b;            % camera → anchor
R_A_C = R_C_A';                   % anchor → camera

%% ── Camera intrinsics ───────────────────────────────────────────────────
fx    = get_cam(cam, 'fx',    620);
fy    = get_cam(cam, 'fy',    620);
cx    = get_cam(cam, 'cx',    480);
cy    = get_cam(cam, 'cy',    270);
W     = get_cam(cam, 'W',     960);
H     = get_cam(cam, 'H',     540);
sigma = get_cam(cam, 'sigma_bearing', 0.0);
rho_min = get_cam(cam, 'rho_min', 0.5);
rho_max = get_cam(cam, 'rho_max', 30.0);

%% ── Pre-allocate ────────────────────────────────────────────────────────
N = map_A.N;
b_f_all = zeros(3, N);
rho_all = zeros(1, N);
eta_all = zeros(1, N);
uvd_all = zeros(3, N);
visible = false(1, N);

%% ── Loop over all features ─────────────────────────────────────────────
for j = 1:N

    %-- Step 1: Range vector in anchor frame
    r_A   = map_A.P(:,j) - p_C_A;         % 3×1  anchor frame

    %-- Step 2: Transform to cameraa frame  (R_A_C = anchor → camera)
    r_C   = R_A_C * r_A;                  % 3×1  camera frame

    %-- Step 3: Depth check — feature must be IN FRONT of camera (Z > 0)
    Z = r_C(3);
    if Z <= 0
        continue
    end

    %-- Step 4: Range check (Euclidean distance)
    d = norm(r_A);
    if d < rho_min || d > rho_max
        continue
    end

    %-- Step 5: Project onto image plane
    u = fx * (r_C(1) / Z) + cx;
    v = fy * (r_C(2) / Z) + cy;

    %-- Step 6: FoV check using image boundary
    if u < 0 || u > W-1 || v < 0 || v > H-1
        continue
    end

    %-- Step 7: Bearing in anchor frame  b_{f,j} = r_A / r_A(3)
    %   By construction b_{f,j}(3) = 1.  ρ = depth along anchor z.
    rho_j = r_A(3);
    if rho_j <= 0, continue; end       % feature must be in front of anchor z
    b_fj  = r_A / rho_j;
    eta_j = norm(b_fj);

    %-- Step 8: Optional angular noise (applied to anchor-frame bearing)
    if sigma > 0
        noise = sigma * randn(2,1);       % noise on b_{f,j}(1:2)
        b_fj(1:2) = b_fj(1:2) + noise;   % b_{f,j}(3) stays 1
        eta_j = norm(b_fj);
    end

    %-- Store
    visible(j)     = true;
    b_f_all(:,j)   = b_fj;
    rho_all(j)     = rho_j;
    eta_all(j)     = eta_j;
    uvd_all(:,j)   = [u; v; Z];

end

%% ── Pack (return dense arrays for sliding window) ───────────────────────
idx_vis = find(visible);
b_f     = b_f_all;                   % 3 × N (0 for invisible)
rho     = rho_all;                   % 1 × N (0 for invisible)
eta     = eta_all;                   % 1 × N (0 for invisible)
uvd     = uvd_all;                   % 3 × N [u; v; Z]

end % compute_bearings

%% ── Helper ──────────────────────────────────────────────────────────────
function val = get_cam(s, field, default)
    if isstruct(s) && isfield(s, field)
        val = s.(field);
    else
        val = default;
    end
end
