function v_body = compute_drone_velocity(v_body, vt, R, ud, params)
%COMPUTE_DRONE_VELOCITY  Rate-limit horizontal velocity and solve altitude hold.
%
%   v_body = compute_drone_velocity(v_body, vt, R, ud, params)
%
%   Rate-limits vx/vy toward the target, then solves for vz so that the
%   NED vertical velocity matches the W/S key command (altitude hold).
%
%   Inputs:
%     v_body — current body-frame velocity [3×1]
%     vt     — target horizontal velocity [3×1]
%     R      — body → world rotation matrix (3×3)
%     ud     — key-state struct (.w, .s for altitude commands)
%     params — struct with: ACCEL, dt, V_MAX, V_ALT
%
%   Output:
%     v_body — updated body-frame velocity [3×1]

ACCEL = params.ACCEL;
dt    = params.dt;
V_MAX = params.V_MAX;
V_ALT = params.V_ALT;

%── Rate-limit vx, vy ────────────────────────────────────────────────────
dv12 = vt(1:2) - v_body(1:2);
v_body(1:2) = v_body(1:2) + min(ACCEL*dt, abs(dv12)) .* sign(dv12);
v_body(1:2) = max(-V_MAX, min(V_MAX, v_body(1:2)));

%── Altitude hold: solve for v_body(3) to cancel vertical NED drift ──────
vz_NED_des = 0;
if ud.w, vz_NED_des = -V_ALT; end   % NED z down → negative = upward
if ud.s, vz_NED_des =  V_ALT; end

r33 = R(3,3);
if abs(r33) > 0.1
    v_body(3) = (vz_NED_des - R(3,1)*v_body(1) - R(3,2)*v_body(2)) / r33;
else
    v_body(3) = vz_NED_des;          % fallback for extreme tilt
end
v_body(3) = max(-V_MAX, min(V_MAX, v_body(3)));

end
