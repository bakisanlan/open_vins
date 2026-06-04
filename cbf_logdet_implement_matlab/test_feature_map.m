%% test_feature_map.m
% =========================================================================
%  Test and visualise create_feature_map + compute_bearings
%  Down-looking camera, intrinsic parameters, pixel-based FoV check.
%
%  Run from the triangulation_VIO root directory.
% =========================================================================
clear; clc;
addpath('utils');
addpath('sim');

fprintf('===== Feature Map + Bearing Tests (Down-looking Camera) =====\n\n');

% ── Camera parameters ────────────────────────────────────────────────────
%  Intrinsics
cam.fx = 620;   cam.fy = 620;
cam.cx = 480;   cam.cy = 270;
cam.W  = 960;   cam.H  = 540;
%  Depth limits
cam.rho_min = 0.5;
cam.rho_max = 60.0;
%  Noise
cam.sigma_bearing = 0.0;   % noiseless for tests
%  Extrinsic: camera → body  (down-looking default, see compute_bearings.m)
cam.R_c2b = [0, -1, 0;
            1,  0, 0;
            0,  0, 1];

% ── Camera FoV half-angles for reference (from intrinsics) ───────────────
fov_h = 2 * atan(cam.W / (2*cam.fx));   % horizontal FoV [rad]
fov_v = 2 * atan(cam.H / (2*cam.fy));   % vertical   FoV [rad]
fprintf('Camera intrinsics: fx=%.0f fy=%.0f cx=%.0f cy=%.0f  W=%dx%d\n', ...
    cam.fx, cam.fy, cam.cx, cam.cy, cam.W, cam.H);
fprintf('Implied FoV: H=%.1f deg  V=%.1f deg\n\n', ...
    rad2deg(fov_h), rad2deg(fov_v));

% ── Feature map ──────────────────────────────────────────────────────────
%  For a down-looking camera at 5 m altitude, features on the ground below.
params.n_features = 60;
params.center     = [0; 0; 0];    % NED: z=0 = ground level
params.extent     = [20; 20; 2];  % flat-ish ground plane ±2 m in z
params.seed       = 42;
map = create_feature_map(params);
fprintf('Feature map: %d landmarks at ground level\n', map.N);

% ── Drone state: 5 m altitude, level, heading North ──────────────────────
p_c = [0; 0; -25];          % 5 m altitude in NED
q   = eul2quat(deg2rad([90,0,0]))';        % identity: body aligned with NED
x   = [p_c; q];

[b_body, b_cam, rho, uvd, idx_vis] = compute_bearings(x, map, cam);
n_vis = numel(idx_vis);
fprintf('Visible features: %d / %d  (at 5 m altitude)\n', n_vis, map.N);
fprintf('Depth range: [%.1f, %.1f] m\n', min(rho), max(rho));

% ── Test 1: All bearing vectors unit length ───────────────────────────────
assert(all(abs(vecnorm(b_body) - 1) < 1e-10), 'FAIL: body bearings not unit');
assert(all(abs(vecnorm(b_cam)  - 1) < 1e-10), 'FAIL: cam  bearings not unit');
fprintf('\nTest 1 — All bearing vectors have unit norm  ✓\n');

% ── Test 2: All pixel projections within image bounds ─────────────────────
assert(all(uvd(1,:) >= 0 & uvd(1,:) <= cam.W-1), 'FAIL: u out of bounds');
assert(all(uvd(2,:) >= 0 & uvd(2,:) <= cam.H-1), 'FAIL: v out of bounds');
fprintf('Test 2 — All pixel projections within [0,%d]x[0,%d]  ✓\n', ...
    cam.W-1, cam.H-1);

% ── Test 3: Known feature directly below drone ────────────────────────────
%   Feature at NED [0;0;0], drone at [0;0;-5] heading North.
%   In camera frame: feature is directly along optical axis (cam +z).
%   Pixel: should be at principal point (cx, cy).
%   Body bearing: cam_z → body_z  (via R_BC), so b_body = [0;0;1].
map_test.P_world = [0; 0; 0];
map_test.N       = 1;
x_test = [[0;0;-5]; [1;0;0;0]];
[b_b_t, b_c_t, rho_t, uvd_t, idx_t] = compute_bearings(x_test, map_test, cam);

assert(~isempty(idx_t),                       'FAIL: feature below not visible');
assert(abs(rho_t - 5) < 1e-10,               'FAIL: depth should be 5 m');
assert(norm(b_c_t - [0;0;1]) < 1e-10,        'FAIL: cam bearing should be [0;0;1]');
assert(norm(b_b_t - [0;0;1]) < 1e-10,        'FAIL: body bearing should be [0;0;1]');
assert(abs(uvd_t(1) - cam.cx) < 1e-8,        'FAIL: u should be cx=480');
assert(abs(uvd_t(2) - cam.cy) < 1e-8,        'FAIL: v should be cy=270');
fprintf('Test 3 — Feature directly below: cam=[0;0;1], body=[0;0;1], pixel=(cx,cy)  ✓\n');

% ── Test 4: Feature offset in East direction ──────────────────────────────
%   Feature at NED [0;2;0] (2 m East, on ground).
%   In NED: East = body +y = cam +x.
%   Expected: u > cx (feature to the right in image).
map_test2.P_world = [0; 2; 0];
map_test2.N       = 1;
[b_b_t2, b_c_t2, ~, uvd_t2, idx_t2] = compute_bearings(x_test, map_test2, cam);
assert(~isempty(idx_t2),      'FAIL: East-offset feature not visible');
assert(uvd_t2(1) > cam.cx,    'FAIL: East feature should project right (u > cx)');
assert(abs(uvd_t2(2) - cam.cy) < 1, 'FAIL: East feature should project near v=cy');
fprintf('Test 4 — East-offset feature projects right of principal point (u=%.1f > cx=%d)  ✓\n', ...
    uvd_t2(1), cam.cx);

fprintf('\n===== All tests PASSED =====\n\n');

% =========================================================================
%  Visualisation 1: 3D scene (NED, z-flipped for readability)
% =========================================================================
figure('Name','Down-Looking Camera — Scene','Color','w','Position',[50 50 1000 500]);

subplot(1,2,1); hold on; grid on; axis equal;
title('3D Scene — NED (z-axis flipped for readability)');
xlabel('East  (NED y) [m]');
ylabel('North (NED x) [m]');
zlabel('Up    (−NED z) [m]');
view(30, 35);

% All features (grey)
P = map.P_world;
plot3(P(2,:), P(1,:), -P(3,:), 'o', ...
    'Color', [0.7 0.7 0.7], 'MarkerFaceColor', [0.7 0.7 0.7], ...
    'MarkerSize', 4, 'DisplayName', 'All features');

% Visible features (blue)
P_vis = map.P_world(:, idx_vis);
plot3(P_vis(2,:), P_vis(1,:), -P_vis(3,:), 'bo', ...
    'MarkerFaceColor', 'b', 'MarkerSize', 6, ...
    'DisplayName', sprintf('Visible (%d)', n_vis));

% Drone
plot3(p_c(2), p_c(1), -p_c(3), 'r^', 'MarkerFaceColor','r', ...
    'MarkerSize', 12, 'DisplayName', 'Drone');

% Bearing rays (blue)
for k = 1:n_vis
    pj = P_vis(:,k);
    plot3([p_c(2), pj(2)], [p_c(1), pj(1)], [-p_c(3), -pj(3)], ...
        'b-', 'LineWidth', 0.5, 'HandleVisibility','off');
end

% Optical axis (camera z → body z → NED z → pointing DOWN = up in flipped plot... show ↓)
R_body = quat2rotm(q');
cam_z_body  = cam.R_c2b * [0;0;1];    % cam optical axis in body frame = [0;0;1]
cam_z_world = R_body * cam_z_body;   % in NED
quiver3(p_c(2), p_c(1), -p_c(3), ...
        cam_z_world(2)*3, cam_z_world(1)*3, -cam_z_world(3)*3, ...
        'r-', 'LineWidth', 2, 'MaxHeadSize', 0.5, 'DisplayName','Optical axis');
legend('Location','best');

% =========================================================================
%  Visualisation 2: Image plane projection
% =========================================================================
subplot(1,2,2); hold on; grid on;
title(sprintf('Image Plane Projection  (%d visible features)', n_vis));
xlabel('u [px]'); ylabel('v [px]');
xlim([0, cam.W]); ylim([0, cam.H]);
pbaspect([cam.W, cam.H, 1]);      % pixel-correct aspect: u-axis wider than v-axis
set(gca,'YDir','reverse');        % image convention: v increases downward

% Image boundary
rectangle('Position', [0, 0, cam.W, cam.H], ...
    'EdgeColor','k', 'LineWidth', 1.5);

% Principal point
plot(cam.cx, cam.cy, 'k+', 'MarkerSize', 12, 'LineWidth', 2, ...
    'DisplayName', sprintf('Principal point (%d,%d)', cam.cx, cam.cy));

% Projected features
scatter(uvd(1,:), uvd(2,:), 40, rho, 'filled', ...
    'DisplayName', 'Features (colour = depth)');
colorbar; colormap('parula');
clim([min(rho), max(rho)]);

legend('Location','best');
fprintf('Figures displayed.\n');
