%% test_vio_observability.m
% =========================================================================
%  VIO-style observability test: track ONE feature across 12 camera poses.
%  Compare 4 scenarios (2x2 grid):
%       Speed  : low (0.5 m/s)  vs  high (2.0 m/s)
%       Altitude: low (2 m)     vs  high (8 m)
%
%  Demonstrates:
%   - Low speed  → coplanar bearings → rank deficiency in M
%   - High speed → diverse bearings  → well-conditioned M
%   - Low alt    → closer feature    → larger 1/rho contribution
%   - High alt   → farther feature   → weaker gradient Psi_v
%
%  Run from triangulation_VIO root.
% =========================================================================
clear; clc;
addpath('utils'); addpath('sim');

%% ── Camera (down-looking) ────────────────────────────────────────────────
cam.fx = 620;  cam.fy = 620;
cam.cx = 480;  cam.cy = 270;
cam.W  = 960;  cam.H  = 540;
cam.rho_min = 0.1;
cam.rho_max = 150.0;
cam.sigma_bearing = 0;
cam.R_c2b = [0,-1,0; 1,0,0; 0,0,1];   % down-looking (cam-z → body-z)

%% ── Single feature in world frame (NED, ground level) ───────────────────
feat_map.P_world = [0; 0; 0];
feat_map.N       = 1;

%% ── Simulation parameters ────────────────────────────────────────────────
N_poses  = 12;
dt       = 0.1;        % [s]
omega_z  = 0.5;        % yaw rate [rad/s] — same for all scenarios

%% ── 2x2 Scenario definitions ─────────────────────────────────────────────
%  {v_fwd [m/s],  altitude [m],  label}
cfg = { 0.5, 2.0, 'Low speed / Low alt';
        0.5, 8.0, 'Low speed / High alt';
        2.0, 2.0, 'High speed / Low alt';
        2.0, 8.0, 'High speed / High alt' };

results = struct();

fprintf('%-28s  %6s  %8s  %8s  %12s\n', ...
    'Scenario','n_vis','logdet','cond','lambda_min');
fprintf('%s\n', repmat('-',1,68));

for s = 1:4
    v_fwd = cfg{s,1};
    h     = cfg{s,2};
    label = cfg{s,3};

    % ── Initial drone state ──────────────────────────────────────────────
    % Centre trajectory over feature: start N/2 steps behind it
    x_start = -(N_poses/2) * dt * v_fwd;   % NED North [m]
    p0      = [x_start; 0; -h];             % NED position (z < 0 = above ground)
    q0      = [1; 0; 0; 0];                 % identity: facing North
    x_state = [p0; q0];

    u = [v_fwd; 0; 0;   0; 0; omega_z];    % forward + yaw (body frame)

    % ── Storage ──────────────────────────────────────────────────────────
    pos_hist    = zeros(3, N_poses);
    q_hist      = zeros(4, N_poses);
    b_world_all = zeros(3, 0);   % bearings rotated to world for sphere plot
    b_body_all  = zeros(3, 0);
    rho_all     = [];
    vis_mask    = false(1, N_poses);

    % ── Integrate N_poses steps ──────────────────────────────────────────
    for k = 1:N_poses
        pos_hist(:,k) = x_state(1:3);
        q_hist(:,k)   = x_state(4:7);

        % Bearing to single feature
        [b_b, ~, rho_k, ~, idx_k] = compute_bearings(x_state, feat_map, cam);

        if ~isempty(idx_k)
            vis_mask(k) = true;
            R_k = quat2rotm(x_state(4:7)');
            b_body_all  = [b_body_all,  b_b];          %#ok<AGROW>
            b_world_all = [b_world_all, R_k * b_b];    %#ok<AGROW>
            rho_all     = [rho_all, rho_k];             %#ok<AGROW>
        end

        % RK4 step
        k1 = drone_kinematics(x_state,           u);
        k2 = drone_kinematics(x_state + dt/2*k1, u);
        k3 = drone_kinematics(x_state + dt/2*k2, u);
        k4 = drone_kinematics(x_state + dt*k3,   u);
        x_state = x_state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
        x_state(4:7) = x_state(4:7) / norm(x_state(4:7));
    end

    % ── Information matrix ───────────────────────────────────────────────
    info = compute_info_matrix(b_body_all, rho_all);

    % ── Store ────────────────────────────────────────────────────────────
    results(s).label       = label;
    results(s).v           = v_fwd;
    results(s).h           = h;
    results(s).pos_hist    = pos_hist;
    results(s).q_hist      = q_hist;
    results(s).b_world     = b_world_all;
    results(s).b_body      = b_body_all;
    results(s).rho         = rho_all;
    results(s).vis_mask    = vis_mask;
    results(s).info        = info;

    fprintf('%-28s  %6d  %8.3f  %8.2f  %12.6f\n', ...
        label, sum(vis_mask), info.logdet, info.cond_num, info.lambda_min);
end

%% =========================================================================
%  FIGURE 1 — 3D Trajectories + Bearing Rays (NED, z-flipped)
%  Rows = speed (low/high), Cols = altitude (low/high)
% =========================================================================
panel_order = [1 2; 3 4];   % scenario index [row, col]
clr_traj    = [0.2 0.5 0.9];
clr_feat    = [0.9 0.2 0.2];

figure('Name','VIO Observability — Trajectories','Color','w',...
    'Position',[50 50 1100 800]);

for row = 1:2
    for col = 1:2
        s   = panel_order(row,col);
        res = results(s);
        P   = res.pos_hist;          % 3×12 NED
        idx = find(res.vis_mask);    % visible pose indices

        ax = subplot(2, 2, (row-1)*2 + col);
        hold on; grid on; axis equal;
        view(30, 35);
        title(res.label, 'FontWeight','bold');
        xlabel('East (y_{NED})'); ylabel('North (x_{NED})'); zlabel('Up (−z_{NED})');

        % Trajectory
        plot3(P(2,:), P(1,:), -P(3,:), '-o', ...
            'Color', clr_traj, 'MarkerFaceColor', clr_traj, ...
            'MarkerSize', 5, 'LineWidth', 1.5, 'DisplayName', 'Trajectory');

        % Number each pose
        for k = 1:N_poses
            text(P(2,k)+0.05, P(1,k)+0.05, -P(3,k)+0.05, ...
                num2str(k), 'FontSize', 7, 'Color', [0.3 0.3 0.3]);
        end

        % Feature
        pf = feat_map.P_world;
        plot3(pf(2), pf(1), -pf(3), 'p', 'MarkerSize', 14, ...
            'MarkerFaceColor', clr_feat, 'MarkerEdgeColor','k', ...
            'DisplayName','Feature');

        % Bearing rays (only visible poses)
        for k = idx
            pc = P(:,k);
            plot3([pc(2), pf(2)], [pc(1), pf(1)], [-pc(3), -pf(3)], ...
                'Color', [0.4 0.8 0.4 0.6], 'LineWidth', 0.8, ...
                'HandleVisibility','off');
        end

        info  = res.info;
        n_vis = sum(res.vis_mask);
        txt = sprintf('n_{vis}=%d/%d\nlog det=%.2f\ncond=%.1f\n\\lambda_{min}=%.4f', ...
            n_vis, N_poses, info.logdet, info.cond_num, info.lambda_min);
        text(ax, 0.02, 0.98, txt, 'Units','normalized', ...
            'VerticalAlignment','top', 'FontSize', 8, ...
            'BackgroundColor', [1 1 0.85], 'EdgeColor', [0.5 0.5 0.5]);

        legend('Location','best','FontSize',7);
    end
end
sgtitle('VIO Observability: Camera Trajectories + Bearing Rays', ...
    'FontSize', 13, 'FontWeight', 'bold');

%% =========================================================================
%  FIGURE 2 — Bearing directions on unit sphere (world frame)
%  Shows visually WHY low-speed gives coplanar / degenerate bearings
% =========================================================================
figure('Name','VIO Observability — Bearing Sphere','Color','w',...
    'Position',[200 50 1100 800]);

for s = 1:4
    res = results(s);
    B   = res.b_world;   % 3×n bearings in world frame

    ax = subplot(2, 2, s);
    hold on; grid on; axis equal;
    view(35, 25);
    xlabel('B_x (N)'); ylabel('B_y (E)'); zlabel('B_z (D)');
    title(res.label, 'FontWeight','bold');

    % Unit sphere (wireframe, transparent)
    [sx,sy,sz] = sphere(24);
    surf(sx, sy, sz, 'FaceAlpha', 0.05, 'EdgeColor', [0.8 0.8 0.8], ...
        'EdgeAlpha', 0.3, 'FaceColor', [0.7 0.7 0.7], 'HandleVisibility','off');

    % Bearing endpoints on sphere (coloured by time index)
    if ~isempty(B)
        scatter3(B(1,:), B(2,:), B(3,:), 60, 1:size(B,2), 'filled', ...
            'MarkerEdgeColor','k', 'DisplayName','Bearing directions');
        colormap(ax, 'cool');
        colorbar(ax, 'Location','eastoutside');
        clim(ax,[1, N_poses]);

        % Draw arrows from centre
        quiver3(zeros(1,size(B,2)), zeros(1,size(B,2)), zeros(1,size(B,2)), ...
            B(1,:), B(2,:), B(3,:), 0, ...
            'Color', [0.3 0.6 0.3], 'LineWidth', 0.8, 'MaxHeadSize', 0.3, ...
            'HandleVisibility','off');
    end

    info = res.info;
    txt = sprintf('log det=%.2f\n\\lambda=[%.3f, %.3f, %.3f]', ...
        info.logdet, info.eigvals(1), info.eigvals(2), info.eigvals(3));
    text(ax, 0.02, 0.98, txt, 'Units','normalized', ...
        'VerticalAlignment','top', 'FontSize', 8, ...
        'BackgroundColor', [1 1 0.85], 'EdgeColor', [0.5 0.5 0.5]);

    xlim([-1 1]); ylim([-1 1]); zlim([-1 1]);
    legend('Location','best','FontSize',7);
end
sgtitle('Bearing Directions on Unit Sphere (World Frame) — Spread = Observability', ...
    'FontSize', 13, 'FontWeight', 'bold');

%% =========================================================================
%  FIGURE 3 — Summary comparison (2x2 metric bar charts)
% =========================================================================
labels   = {results.label};
logdets  = arrayfun(@(r) r.info.logdet,     results);
conds    = arrayfun(@(r) r.info.cond_num,   results);
lmins    = arrayfun(@(r) r.info.lambda_min, results);
n_vis_v  = arrayfun(@(r) sum(r.vis_mask),   results);
colors   = [0.3 0.6 0.9; 0.3 0.9 0.6; 0.9 0.6 0.3; 0.9 0.3 0.6];

figure('Name','VIO Observability — Metrics Summary','Color','w',...
    'Position',[350 50 900 600]);

metrics      = {logdets;  lmins;  1./max(conds,1e-3);  n_vis_v};
metric_names = {'log det(M)', '\lambda_{min}(M)', '1/cond(M)  (higher=better)', ...
                'N visible / 12'};

for m = 1:4
    subplot(2,2,m);
    vals = metrics{m}';
    b    = bar(vals, 'FaceColor','flat');
    for k = 1:4, b.CData(k,:) = colors(k,:); end
    set(gca, 'XTickLabel', {'LS/LA','LS/HA','HS/LA','HS/HA'}, ...
        'XTickLabelRotation', 15, 'FontSize', 9);
    ylabel(metric_names{m});
    title(metric_names{m}, 'FontWeight','bold');
    grid on; yline(0,'k--','LineWidth',0.8);
end
sgtitle({'Metric Summary (LS=Low Speed, HS=High Speed, LA=Low Alt, HA=High Alt)';
         'High speed + Low altitude provides best observability'}, ...
    'FontSize', 11, 'FontWeight','bold');

fprintf('\nKey insight:\n');
fprintf('  log det(M): %.3f | %.3f | %.3f | %.3f\n', logdets);
fprintf('  Higher log det = better conditioned M = more observable triangulation\n');
