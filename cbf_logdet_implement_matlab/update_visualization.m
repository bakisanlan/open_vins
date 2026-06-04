function trail_d = update_visualization(vis, sim_state, trail_d, ld_hist, info, params)
%UPDATE_VISUALIZATION  Update all three display panels each simulation tick.
%
%   trail_d = update_visualization(vis, sim_state, trail_d, ld_hist, info, params)
%
%   Panels updated:
%     1. 3D scene — trail, drone marker, arms, body axes, bearing line, HUD
%     2. Log-det  — history curve, current marker, safety text, auto-scale
%     3. Attitude — body axes, arm cross, RPY readout
%
%   Inputs:
%     vis       — handle struct from setup_visualization
%     sim_state — struct with fields:
%                   .x, .R, .v_safe, .om, .rpy, .idx_k,
%                   .cbf_active, .CBF_ON, .win_size, .gamma
%     trail_d   — [3×N] display-frame trail positions
%     ld_hist   — [N_HIST×1] log-det history
%     info      — struct from compute_info_matrix
%     params    — struct with: ARM, AX_SC, W_WIN, h_min, TRAIL, pf_d
%
%   Output:
%     trail_d   — updated trail (with current position appended, trimmed)

%── Unpack state ──────────────────────────────────────────────────────────
x          = sim_state.x;
R          = sim_state.R;
v_body     = sim_state.v_body;
v_safe     = sim_state.v_safe;
om         = sim_state.om;
rpy        = sim_state.rpy;
idx_k      = sim_state.idx_k;
cbf_active = sim_state.cbf_active;
CBF_ON     = sim_state.CBF_ON;
win_size   = sim_state.win_size;
gamma_val  = sim_state.gamma;

ARM   = params.ARM;
AX_SC = params.AX_SC;
W_WIN = params.W_WIN;
h_min = params.h_min;
TRAIL = params.TRAIL;
pf_d  = params.pf_d;

%% ═══════════════════════ DISPLAY COORD TRANSFORMS ════════════════════════
pc  = x(1:3);
pd  = n2d(pc);                        % NED → ENU display
alt = -pc(3);

trail_d = [trail_d, pd];
if size(trail_d,2) > TRAIL, trail_d = trail_d(:,end-TRAIL+1:end); end

Rd  = @(v) n2d(R*v);                  % body axis → display frame
Xd  = Rd([1;0;0]);  Yd = Rd([0;1;0]);  Zd = Rd([0;0;1]);

arm_f = pd + Xd*ARM;  arm_b = pd - Xd*ARM;
arm_r = pd + Yd*ARM;  arm_l = pd - Yd*ARM;
bx_e  = pd + Xd*AX_SC;
by_e  = pd + Yd*AX_SC;
bz_e  = pd + Zd*AX_SC;

%% ═══════════════════════ 3D SCENE PANEL ══════════════════════════════════

% Trail + drone marker
set(vis.htr,'XData',trail_d(1,:),'YData',trail_d(2,:),'ZData',trail_d(3,:));
set(vis.hdr,'XData',pd(1),'YData',pd(2),'ZData',pd(3));

% Drone arms
set(vis.ha1,'XData',[arm_b(1) arm_f(1)],'YData',[arm_b(2) arm_f(2)],'ZData',[arm_b(3) arm_f(3)]);
set(vis.ha2,'XData',[arm_l(1) arm_r(1)],'YData',[arm_l(2) arm_r(2)],'ZData',[arm_l(3) arm_r(3)]);

% Body-axis arrows
set(vis.hbX,'XData',[pd(1) bx_e(1)],'YData',[pd(2) bx_e(2)],'ZData',[pd(3) bx_e(3)]);
set(vis.hbY,'XData',[pd(1) by_e(1)],'YData',[pd(2) by_e(2)],'ZData',[pd(3) by_e(3)]);
set(vis.hbZ,'XData',[pd(1) bz_e(1)],'YData',[pd(2) bz_e(2)],'ZData',[pd(3) bz_e(3)]);

% Velocity arrows
V_SC = 0.5;  % Scale factor for velocity arrows
Vd_nom = Rd(v_body);
Vd_safe = Rd(v_safe);
vnom_e = pd + Vd_nom * V_SC;
vsafe_e = pd + Vd_safe * V_SC;

if norm(v_body) > 0.1
    set(vis.hVnom,'XData',[pd(1) vnom_e(1)],'YData',[pd(2) vnom_e(2)],'ZData',[pd(3) vnom_e(3)]);
else
    set(vis.hVnom,'XData',[nan nan],'YData',[nan nan],'ZData',[nan nan]);
end

if norm(v_safe) > 0.1
    set(vis.hVsafe,'XData',[pd(1) vsafe_e(1)],'YData',[pd(2) vsafe_e(2)],'ZData',[pd(3) vsafe_e(3)]);
else
    set(vis.hVsafe,'XData',[nan nan],'YData',[nan nan],'ZData',[nan nan]);
end

% Bearing line (green = normal, orange = CBF active)
if ~isempty(idx_k)
    n_vis = length(idx_k);
    XData = nan(1, 3*n_vis); YData = nan(1, 3*n_vis); ZData = nan(1, 3*n_vis);
    XData(1:3:end) = pd(1); XData(2:3:end) = pf_d(1, idx_k);
    YData(1:3:end) = pd(2); YData(2:3:end) = pf_d(2, idx_k);
    ZData(1:3:end) = pd(3); ZData(2:3:end) = pf_d(3, idx_k);
    
    if cbf_active
        set(vis.hbr,'XData',XData,'YData',YData,'ZData',ZData,'Color',[1.0 0.6 0.1]);
    else
        set(vis.hbr,'XData',XData,'YData',YData,'ZData',ZData,'Color',[0.2 0.9 0.4]);
    end
else
    set(vis.hbr,'XData',[nan nan],'YData',[nan nan],'ZData',[nan nan]);
end

% Auto-fit axis limits
MARGIN = 3.0;
pts    = [pd, pf_d];
lo     = min(pts, [], 2) - MARGIN;
hi     = max(pts, [], 2) + MARGIN;
extent = max(hi - lo) / 2;
ctr    = (lo + hi) / 2;
xlim(vis.a3, [ctr(1)-extent, ctr(1)+extent]);
ylim(vis.a3, [ctr(2)-extent, ctr(2)+extent]);
zlim(vis.a3, [ctr(3)-extent, ctr(3)+extent]);

% HUD text
vspd        = norm(v_safe);
cbf_str     = ternary(CBF_ON, 'ON ', 'OFF');
cbf_act_str = ternary(cbf_active, ' CORRECTING', '');
clr_ld      = ternary(info.logdet >= h_min, ' SAFE  ', 'UNSAFE ');
hud = sprintf([ ...
    'Alt  : %5.1f m\n' ...
    'Pos  : N%5.1f E%5.1f\n' ...
    'Speed: %5.2f m/s\n' ...
    'Vx   : %5.2f  Vy: %5.2f  Vz: %5.2f\n' ...
    'Wx   : %5.2f  Wy: %5.2f  Wz: %5.2f\n' ...
    'n_win: %2d / %2d bearings\n' ...
    'logdet: %6.3f  [%s]\n' ...
    'λ_min: %7.4f  cond: %.1f\n' ...
    'CBF  : %s  γ=%.1f%s'], ...
    alt, pc(1), pc(2), vspd, ...
    v_safe(1), v_safe(2), v_safe(3), ...
    om(1), om(2), om(3), ...
    win_size, W_WIN, ...
    info.logdet, clr_ld, ...
    info.lambda_min, info.cond_num, ...
    cbf_str, gamma_val, cbf_act_str);
set(vis.hHUD, 'String', hud);

%% ═══════════════════════ LOG-DET PANEL ═══════════════════════════════════

set(vis.hLD,  'YData', ld_hist);
set(vis.hLDn, 'YData', info.logdet);

if ~isnan(info.logdet)
    safe_str = ternary(info.logdet >= h_min, '✓ SAFE', '✗ UNSAFE');
    set(vis.hLDt, 'String', sprintf('log det = %.3f  %s', info.logdet, safe_str), ...
        'Color', ternary(info.logdet >= h_min, [0.3 0.9 0.5], [0.9 0.3 0.3]));
end

% Auto-scale y-axis
visible_vals = ld_hist(isfinite(ld_hist));
if numel(visible_vals) > 2
    lo_y = min(visible_vals) - 0.5;
    hi_y = max(visible_vals) + 0.5;
    if hi_y > lo_y, ylim(vis.ald, [lo_y, hi_y]); end
end

%% ═══════════════════════ ATTITUDE PANEL ══════════════════════════════════

set(vis.haX, 'XData',[0 Xd(1)],'YData',[0 Xd(2)],'ZData',[0 Xd(3)]);
set(vis.haY, 'XData',[0 Yd(1)],'YData',[0 Yd(2)],'ZData',[0 Yd(3)]);
set(vis.haZ, 'XData',[0 Zd(1)],'YData',[0 Zd(2)],'ZData',[0 Zd(3)]);
set(vis.haA1,'XData',[-Xd(1) Xd(1)],'YData',[-Xd(2) Xd(2)],'ZData',[-Xd(3) Xd(3)]);
set(vis.haA2,'XData',[-Yd(1) Yd(1)],'YData',[-Yd(2) Yd(2)],'ZData',[-Yd(3) Yd(3)]);
set(vis.hRPY,'String',sprintf('R:%+5.1f°  P:%+5.1f°  Y:%+5.1f°', ...
    rad2deg(rpy(3)), rad2deg(rpy(2)), rad2deg(rpy(1))));

end

%% ══════════════════════════ LOCAL HELPERS ══════════════════════════════════

function d = n2d(p)
% NED → ENU-visual display (x=East, y=North, z=Up)
d = [p(2,:); p(1,:); -p(3,:)];
end

function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end
