%% test_kinematics.m
% =========================================================================
%  Sanity-check tests for drone_kinematics.m and quat2rot.m
%
%  Tests performed:
%   1. Identity quaternion → position moves along x (North) correctly
%   2. 90° yaw → body x maps to world y (East)
%   3. Pure yaw rate → quaternion evolves correctly
%   4. RK4 integration: circular horizontal trajectory
%
%  Run from the triangulation_VIO root (adds utils/ and sim/ to path).
% =========================================================================

clear; clc;
addpath('utils');
addpath('sim');

fprintf('===== Drone Kinematics Sanity Checks =====\n\n');

% -------------------------------------------------------------------------
%  Test 1: Identity quaternion, move forward (+North)
% -------------------------------------------------------------------------
fprintf('--- Test 1: Identity orientation, v = [1;0;0] body ---\n');
q_id = [1;0;0;0];
x0   = [0;0;0; q_id];          % at origin, level
u    = [1;0;0; 0;0;0];          % 1 m/s forward (body x = North)
xd   = drone_kinematics(x0, u);
fprintf('  p_dot (expected [1;0;0] NED): [%.3f; %.3f; %.3f]\n', xd(1:3));
assert(norm(xd(1:3) - [1;0;0]) < 1e-10, 'Test 1 FAILED');
fprintf('  PASSED\n\n');

% -------------------------------------------------------------------------
%  Test 2: 90 deg yaw (CW from above in NED), move forward
%          Body x should map to world y (East) after 90 deg yaw
% -------------------------------------------------------------------------
fprintf('--- Test 2: 90 deg yaw, v = [1;0;0] body ---\n');
psi = pi/2;                           % 90 deg CW yaw in NED
%  Rotation about NED z-axis (down) by +psi:
%  q = [cos(psi/2); 0; 0; sin(psi/2)]
q_yaw = [cos(psi/2); 0; 0; sin(psi/2)];
x0    = [0;0;0; q_yaw];
u     = [1;0;0; 0;0;0];
xd    = drone_kinematics(x0, u);
fprintf('  p_dot (expected [0;1;0] NED = East): [%.3f; %.3f; %.3f]\n', xd(1:3));
assert(norm(xd(1:3) - [0;1;0]) < 1e-10, 'Test 2 FAILED');
fprintf('  PASSED\n\n');

% -------------------------------------------------------------------------
%  Test 3: Pure yaw rate, quaternion should spin about z-axis
% -------------------------------------------------------------------------
fprintf('--- Test 3: Pure yaw rate om = [0;0;1] rad/s ---\n');
q_id = [1;0;0;0];
x0   = [0;0;0; q_id];
u    = [0;0;0; 0;0;1];              % 1 rad/s yaw (om_z body = om_z NED at identity)
xd   = drone_kinematics(x0, u);
% Expected q_dot = 0.5 * Xi([1;0;0;0]) * [0;0;1]
% Xi([1,0,0,0]) = [0,0,0; 1,0,0; 0,1,0; 0,0,1]
% q_dot = 0.5 * [0;0;0;1]
fprintf('  q_dot (expected 0.5*[0;0;0;1]): [%.4f; %.4f; %.4f; %.4f]\n', xd(4:7));
assert(norm(xd(4:7) - 0.5*[0;0;0;1]) < 1e-10, 'Test 3 FAILED');
fprintf('  PASSED\n\n');

% -------------------------------------------------------------------------
%  Test 4: RK4 integration — horizontal circle in NED plane
%          Constant forward speed + yaw rate → should trace a circle
% -------------------------------------------------------------------------
fprintf('--- Test 4: Circular trajectory integration (RK4) ---\n');

dt   = 0.001;                  % time step [s]
T    = 2*pi;                  % one full circle [s]
N    = round(T/dt);
v    = 1.0;                   % forward speed [m/s]
om_z = 1.0;                   % yaw rate [rad/s]  →  circle radius = v/om_z = 1 m

x    = [0;0;0; 1;0;0;0];     % start at origin, heading North
traj = zeros(3, N+1);
traj(:,1) = x(1:3);

u = [v; 0; 0;   0; 0; om_z];  % constant input

for k = 1:N
    % RK4 step
    k1 = drone_kinematics(x,           u);
    k2 = drone_kinematics(x + dt/2*k1, u);
    k3 = drone_kinematics(x + dt/2*k2, u);
    k4 = drone_kinematics(x + dt*k3,   u);
    x  = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
    x(4:7) = x(4:7) / norm(x(4:7));   % re-normalise quaternion
    traj(:,k+1) = x(1:3);
end

% After one full revolution, position should return near start
pos_error = norm(traj(:,end) - traj(:,1));
fprintf('  Position error after 2π revolution: %.4f m (expect ~0)\n', pos_error);
assert(pos_error < 0.01, 'Test 4 FAILED: trajectory did not close');
fprintf('  PASSED\n\n');

% Plot the trajectory
figure('Name','Kinematics Test 4 — Circular Trajectory','Color','w');
plot(traj(2,:), traj(1,:), 'b-', 'LineWidth', 2);  % East vs North
hold on;
plot(traj(2,1), traj(1,1), 'go', 'MarkerSize', 10, 'LineWidth', 2);
plot(traj(2,end), traj(1,end), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
axis equal; grid on;
xlabel('East (y_{NED}) [m]');
ylabel('North (x_{NED}) [m]');
title('Circular Trajectory — NED Top View (z=const)');
legend('Trajectory','Start','End','Location','best');

fprintf('===== All tests PASSED =====\n');
fprintf('NED frame conventions confirmed:\n');
fprintf('  Body x → North (NED x)  at zero yaw\n');
fprintf('  Body y → East  (NED y)  at zero yaw\n');
fprintf('  Body z → Down  (NED z)  at zero yaw\n');
fprintf('  Positive yaw rate → clockwise from above (NED)\n');
