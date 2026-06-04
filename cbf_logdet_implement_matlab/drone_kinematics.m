function xdot = drone_kinematics(x, u)
%DRONE_KINEMATICS  Kinematic model of a rigid-body drone in NED/FRD frames.
%
%   xdot = drone_kinematics(x, u)
%
%   This is a KINEMATIC model (velocity-controlled). It is appropriate for
%   testing a velocity-level CBF controller before adding full dynamics.
%
% =========================================================================
%  Frame Conventions
% =========================================================================
%
%   World frame  : NED (North-East-Down), right-handed
%                  x → North,  y → East,  z → Down
%
%   Body frame   : FRD (Forward-Right-Down), right-handed
%                  x → Forward,  y → Right,  z → Down
%
%   Quaternion   : Hamilton convention, scalar-first  q = [q0;q1;q2;q3]
%                  Represents rotation FROM body (FRD) TO world (NED)
%                  i.e.   v_NED = R(q) * v_body
%
% =========================================================================
%  State
% =========================================================================
%
%   x = [p_c (3x1)]   camera/drone position in NED world frame [m]
%       [q   (4x1)]   unit quaternion body→world [Hamilton, scalar first]
%
%   Total: 7 states
%
% =========================================================================
%  Control Input
% =========================================================================
%
%   u = [v   (3x1)]   linear  velocity in BODY (FRD) frame [m/s]
%       [om  (3x1)]   angular velocity in BODY (FRD) frame [rad/s]
%
%   Total: 6 inputs  (this is what the CBF-QP outputs)
%
% =========================================================================
%  Kinematics
% =========================================================================
%
%   Position:   p_dot = R(q) * v_body
%
%   Quaternion: q_dot = 0.5 * Xi(q) * om_body
%
%   where Xi(q) is the 4x3 right-multiplication matrix:
%
%       Xi(q) = [ -q1  -q2  -q3 ]
%               [  q0  -q3   q2 ]
%               [  q3   q0  -q1 ]
%               [ -q2   q1   q0 ]
%
%   Derived from the Hamilton product:  q_dot = 0.5 * (q ⊗ [0; om_body])
%
% =========================================================================
%  Inputs & Outputs
% =========================================================================
%
%   Inputs:
%       x  - 7x1 state vector  [p_c(1:3); q(4:7)]
%       u  - 6x1 control input [v_body(1:3); om_body(4:6)]
%
%   Output:
%       xdot - 7x1 state derivative [p_c_dot(1:3); q_dot(4:7)]
%
% =========================================================================

%% --- Unpack state ---
p_c = x(1:3);   % position in NED [m]            (unused in derivative
                 % directly, but kept for consistency)             %#ok<NASGU>
q   = x(4:7);   % quaternion [q0;q1;q2;q3]

%% --- Unpack control ---
v_b  = u(1:3);   % linear  velocity, body frame [m/s]
om_b = u(4:6);   % angular velocity, body frame [rad/s]

%% --- Normalise quaternion (defend against integration drift) ---
q = q / norm(q);
q0 = q(1);  q1 = q(2);  q2 = q(3);  q3 = q(4);

%% --- Rotation matrix R: body (FRD) → world (NED) ---
%   v_NED = R * v_body
R = quat2rotm(q');   % 3x3, MATLAB built-in (needs row vector input)

%% --- Position derivative (NED) ---
%   p_dot = R * v_body
p_dot = R * v_b;   % 3x1

%% --- Quaternion kinematic matrix Xi(q), shape 4x3 ---
%   q_dot = 0.5 * Xi(q) * om_body
%
%   Derivation:  q_dot = 0.5 * (q ⊗ [0; om_body])
%   For Hamilton product row by row:
%       q_dot0 = -q1*wx - q2*wy - q3*wz
%       q_dot1 =  q0*wx - q3*wy + q2*wz     (wait: check sign)
%       q_dot2 =  q3*wx + q0*wy - q1*wz     (check)
%       q_dot3 = -q2*wx + q1*wy + q0*wz     (check)
%
% Xi = [-q1, -q2, -q3;
%        q0, -q3,  q2;
%        q3,  q0, -q1;
%       -q2,  q1,  q0];   % 4x3
% 
% q_dot = 0.5 * Xi * om_b;   % 4x1

r = [0 ; om_b];
q_dot = 0.5 * quatmultiply(q',r')';

%% --- Assemble output ---
xdot = [p_dot; q_dot];   % 7x1

end % drone_kinematics
