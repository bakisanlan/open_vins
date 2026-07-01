function [vt, om, rpy, ud] = process_keyboard_input(fig, x, params)
%PROCESS_KEYBOARD_INPUT  Read keys and compute velocity targets + angular rates.
%
%   [vt, om, rpy, ud] = process_keyboard_input(fig, x, params)
%
%   Reads the current key-state from figure appdata and converts it into:
%     - Target horizontal velocity (body frame)
%     - Angular rate commands (with auto-leveling when keys released)
%
%   Inputs:
%     fig    — figure handle (keys stored via appdata)
%     x      — state vector [pos(3); quat(4)]
%     params — struct with: V_PITCH, V_ROLL, W_P, W_R, W_Y, K_LEVEL
%
%   Outputs:
%     vt   — target horizontal velocity [3×1] (body frame, vz=0)
%     om   — angular rate command [3×1] (body frame)
%     rpy  — current Euler angles [yaw, pitch, roll] (rad)
%     ud   — key-state struct (needed downstream for altitude hold)

V_PITCH = params.V_PITCH;
V_ROLL  = params.V_ROLL;
W_P     = params.W_P;
W_R     = params.W_R;
W_Y     = params.W_Y;
K_LEVEL = params.K_LEVEL;

%── Read keys ─────────────────────────────────────────────────────────────
ud = getappdata(fig, 'keys');

%── Target horizontal velocity ────────────────────────────────────────────
vt = zeros(3,1);
if ud.uparrow,    vt(1) =  V_PITCH; end
if ud.downarrow,  vt(1) = -V_PITCH; end
if ud.rightarrow, vt(2) =  V_ROLL;  end
if ud.leftarrow,  vt(2) = -V_ROLL;  end

%── Current Euler angles (ZYX: yaw-pitch-roll) ────────────────────────────
rpy = quat2eul(x(4:7)', 'ZYX');   % [yaw, pitch, roll] in rad

%── Angular rates (direct from keys + auto-leveling on release) ───────────
om = zeros(3,1);

if ud.uparrow                         % pitch down
    om(2) = om(2) - W_P;
elseif ud.downarrow                    % pitch up
    om(2) = om(2) + W_P;
else                                   % auto-level pitch
    om(2) = -K_LEVEL * rpy(2);
end

if ud.rightarrow                       % roll right
    om(1) = om(1) + W_R;
elseif ud.leftarrow                    % roll left
    om(1) = om(1) - W_R;
else                                   % auto-level roll
    om(1) = -K_LEVEL * rpy(3);
end

if ud.d,  om(3) = om(3) + W_Y; end    % yaw right
if ud.a,  om(3) = om(3) - W_Y; end    % yaw left

end
