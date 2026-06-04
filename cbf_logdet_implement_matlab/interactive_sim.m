function interactive_sim()
%INTERACTIVE_SIM  Keyboard-controlled drone + real-time observability display.
%
%  Controls (click figure to give it keyboard focus):
%    W / S          : altitude up / down        (no rotation)
%    UP / DOWN      : pitch down/up  → fwd/bwd velocity + pitch rotation
%    RIGHT / LEFT   : roll right/left → ±Y velocity + roll rotation
%    D / A          : yaw right / left
%    C              : toggle CBF safety filter ON/OFF
%    Q or ESC       : quit
%
%  Anchor-frame architecture:
%    All bearings and the info matrix are computed in an anchor frame A,
%    defined as a camera pose in the sliding window.  When the anchor is
%    evicted from the window, the next-oldest visible camera pose becomes
%    the new anchor and all stored quantities are re-expressed accordingly.
%
%  Run: >> interactive_sim()   (from triangulation_VIO root)

addpath('utils'); addpath('sim');

%% ── Tunable parameters ─────────────────────────────────────────────────
params = struct( ...
    'V_MAX',   15.0, ...   % max body-frame speed [m/s]
    'ACCEL',    2.0, ...   % velocity ramp rate [m/s²]
    'V_ALT',    1.0, ...   % altitude ΔV [m/s]
    'V_PITCH', 48.0, ...   % max fwd/bwd speed from pitch [m/s]
    'V_ROLL',  48.0, ...   % max lateral speed from roll [m/s]
    'W_P',      0.2, ...   % pitch rate [rad/s]
    'W_R',      0.2, ...   % roll  rate [rad/s]
    'W_Y',      1.0, ...   % yaw   rate [rad/s]
    'K_LEVEL',  2.0, ...   % auto-level gain [rad/s per rad of error]
    'dt',       0.05, ...  % sim timestep [s]
    'W_WIN',   12, ...     % bearing sliding-window size
    'N_HIST', 200, ...     % log-det history length
    'TRAIL',   80, ...     % trail length
    'h_min',    0.5, ...   % log-det safety threshold
    'ARM',      0.4, ...   % drone arm visual length [m]
    'AX_SC',    0.8);      % body-axis arrow length [m]

T = 0;

%% ── CBF parameters ─────────────────────────────────────────────────────
CBF_ON  = false;           % toggled with 'C' key
gamma   = 1.0;             % class-K decay rate  (alpha(h)=gamma*h)

%% ── Camera (down-looking) ──────────────────────────────────────────────
cam.fx=620; cam.fy=620; cam.cx=480; cam.cy=270;
cam.W=960;  cam.H=540;  cam.rho_min=0.2; cam.rho_max=150;
cam.sigma_bearing=0;
cam.R_c2b=[0,-1,0; 1,0,0; 0,0,1];

%% ── Multiple ground features (Uniform grid) ───────────────────────────
[X, Y] = meshgrid(-20:10:20, -20:10:20);
feat_W = [X(:)'; Y(:)'; zeros(1, numel(X))];
params.pf_d = n2d(feat_W);                % display coords (fixed)

%% ── Drone world-frame state (used for kinematics only) ────────────────
x       = [0;-8;-100; 1;0;0;0];             % pos=[0,-8,-5] NED, level quat
v_body  = zeros(3,1);

%% ── Anchor frame state ─────────────────────────────────────────────────
%   p_A_W  : anchor position in world  (needed to convert drone state → anchor)
%   R_A_W  : anchor → world rotation   (camera frame at anchor time)
%   p_f_A  : feature position in current anchor frame
%   map_A  : struct passed to compute_bearings
anchor_valid = false;
p_A_W = zeros(3,1);
R_A_W = eye(3);
p_f_A = zeros(3,1);
map_A = struct('P', zeros(3,size(feat_W,2)), 'N', size(feat_W,2));

%% ── Sliding window ─────────────────────────────────────────────────────
%   Per entry: anchor-frame bearing, depth, norm, camera→anchor rotation,
%   and camera position in anchor frame (needed for re-anchoring).
win_bf    = zeros(3,0,size(feat_W,2));  % bearings in anchor frame
win_rho   = zeros(1,0,size(feat_W,2));  % depths
win_eta   = zeros(1,0,size(feat_W,2));  % norms
win_RCjA  = zeros(3,3,0);               % R_{C_j}^A  per frame
win_pCjA  = zeros(3,0);                 % camera-j position in anchor frame per frame
anchor_win_idx = 1;                     % index of the anchor in the sliding window

logdet_hist    = nan(params.N_HIST, 1);
trail_d    = zeros(3,0);
running    = true;

%% ── Create visualization ──────────────────────────────────────────────
[fig, vis] = setup_visualization(params);
set(fig, 'KeyPressFcn', @kp, 'KeyReleaseFcn', @kr, ...
    'CloseRequestFcn', @on_close);
setappdata(fig, 'keys', false_keys());
setappdata(fig, 'quit', false);

%% ═══════════════════════════ MAIN LOOP ═══════════════════════════════════
while running && ishandle(fig) && ~getappdata(fig,'quit')
    t0 = tic;

    %── 1. Current world-frame rotation ───────────────────────────────────
    R = quat2rotm(x(4:7)');

    %── 2. Convert current pose to anchor frame ───────────────────────────
    if ~anchor_valid
        % No anchor yet — tentatively use current camera as anchor
        R_A_W = R * cam.R_c2b;                        % camera→world = anchor→world
        p_A_W = x(1:3);
        p_f_A = R_A_W' * (feat_W - p_A_W);            % feature in tentative anchor
        map_A.P = p_f_A;  map_A.N = size(feat_W, 2);
    end
    p_C_A = R_A_W' * (x(1:3) - p_A_W);                % camera pos in anchor frame
    R_B_A = R_A_W' * R;                                % body → anchor

    %── 3. Compute bearings in anchor frame ───────────────────────────────
    [b_f_k, rho_k, eta_k, ~, idx_k, R_C_A_k] = ...
        compute_bearings(p_C_A, R_B_A, map_A, cam);

    disp(b_f_k)
    % disp(T);

    %── 3b. Re-acquire feature with fresh anchor if stale ─────────────────
    %   The bearing parameterisation requires rho_j = r_A(3) > 0, which
    %   uses the ANCHOR z-axis.  When the anchor is stale (feature was lost
    %   and no re-anchoring occurred), this check can reject a feature that
    %   IS visible in the current camera.  Fix: retry with a fresh anchor
    %   at the current camera pose and, if the feature is found, reset.
    if anchor_valid && isempty(idx_k)
        R_A_try = R * cam.R_c2b;
        p_A_try = x(1:3);
        pf_try  = R_A_try' * (feat_W - p_A_try);
        map_try = struct('P', pf_try, 'N', size(feat_W, 2));
        [b_f_k, rho_k, eta_k, ~, idx_k, R_C_A_k] = ...
            compute_bearings(zeros(3,1), R_A_try' * R, map_try, cam);
        if ~isempty(idx_k)
            % Feature re-acquired — adopt fresh anchor, flush stale window
            R_A_W = R_A_try;
            p_A_W = p_A_try;
            p_f_A = pf_try;
            map_A = map_try;
            p_C_A = zeros(3,1);
            R_B_A = R_A_W' * R;
            win_bf = zeros(3,0,size(feat_W,2));  win_rho = zeros(1,0,size(feat_W,2));  win_eta = zeros(1,0,size(feat_W,2));
            win_RCjA = zeros(3,3,0);  win_pCjA = zeros(3,0);
            anchor_win_idx = 1;
        end
    end

    %── 4. Set anchor on first visible observation ────────────────────────
    if ~anchor_valid && ~isempty(idx_k)
        anchor_valid = true;
    end

    %── 5. Push to sliding window ─────────────────────────────────────────
    if anchor_valid && ~isempty(idx_k)
        win_bf   = cat(2, win_bf, reshape(b_f_k, [3, 1, size(b_f_k,2)]));
        win_rho  = cat(2, win_rho, reshape(rho_k, [1, 1, size(rho_k,2)]));
        win_eta  = cat(2, win_eta, reshape(eta_k, [1, 1, size(eta_k,2)]));
        win_RCjA = cat(3, win_RCjA, R_C_A_k);
        win_pCjA = cat(2, win_pCjA, p_C_A);
    end

    %── 6. Enforce window size ────────────────────────────────────────────
    %   Re-anchor to the NEWEST camera pose ONLY when the anchor is about
    %   to be marginalized (evicted) from the sliding window.
    if size(win_bf, 2) > params.W_WIN
        if anchor_win_idx == 1
            reanchor_to(size(win_bf, 2));   % new anchor = newest camera (last slot)
            anchor_win_idx = size(win_bf, 2);
            disp('reanchoring')
            disp(T);
        end
        
        win_bf   = win_bf(:,   2:end, :);
        win_rho  = win_rho(:,  2:end, :);
        win_eta  = win_eta(:,  2:end, :);
        win_RCjA = win_RCjA(:,:,2:end);
        win_pCjA = win_pCjA(:, 2:end);
        
        anchor_win_idx = anchor_win_idx - 1;
    end

    %── 7. Compute info matrix ────────────────────────────────────────────
    info    = compute_info_matrix(win_bf, win_rho, win_eta, win_RCjA, cam.R_c2b, length(idx_k));
    logdet_hist = [logdet_hist(2:end); info.logdet];

    %── 8. Process keyboard input ─────────────────────────────────────────
    [vt, om, rpy, ud] = process_keyboard_input(fig, x, params);

    %── 9. Compute nominal target velocity (rate-limit + altitude hold) ───
    v_prev = v_body;
    v_nom  = compute_drone_velocity(v_body, vt, R, ud, params);

    %── 10. CBF safety filter ─────────────────────────────────────────────
    [v_safe_unbounded, cbf_active] = compute_cbf_correction( ...
        v_nom, info, CBF_ON, gamma, params.h_min, R_B_A, anchor_valid);

    %── 11. Apply rate and absolute limits to CBF commands ────────────────
    dv = v_safe_unbounded - v_prev;
    v_safe = v_prev + min(params.ACCEL*params.dt, abs(dv)) .* sign(dv);
    v_safe = max(-params.V_MAX, min(params.V_MAX, v_safe));
    
    v_body = v_safe;

    %── 11. RK4 integration (world frame) ─────────────────────────────────
    u  = [v_safe; om];
    f1 = drone_kinematics(x, u);
    f2 = drone_kinematics(x + params.dt/2*f1, u);
    f3 = drone_kinematics(x + params.dt/2*f2, u);
    f4 = drone_kinematics(x + params.dt*f3, u);
    x  = x + (params.dt/6)*(f1 + 2*f2 + 2*f3 + f4);
    x(4:7) = x(4:7) / norm(x(4:7));

    %── 12. Update visualization ──────────────────────────────────────────
    sim_state = struct( ...
        'x', x, 'R', R, 'v_body', v_body, 'v_safe', v_safe, 'om', om, ...
        'rpy', rpy, 'idx_k', idx_k, 'cbf_active', cbf_active, ...
        'CBF_ON', CBF_ON, 'win_size', size(win_bf,2), 'gamma', gamma);
    trail_d = update_visualization(vis, sim_state, trail_d, logdet_hist, info, params);

    drawnow limitrate;

    % Timing hold
    elapsed = toc(t0);
    T = T + params.dt;
    if elapsed < params.dt, pause(params.dt - elapsed); end
end

delete(fig);
fprintf('Simulation ended.\n');

%% ══════════════════════════ NESTED FUNCTIONS ══════════════════════════════

    function kp(~, evt)
        ud_kp = getappdata(fig,'keys');
        k  = lower(strrep(evt.Key,' ',''));
        if isfield(ud_kp,k), ud_kp.(k) = true; setappdata(fig,'keys',ud_kp); end
        if strcmp(k,'q') || strcmp(k,'escape'), setappdata(fig,'quit',true); end
        if strcmp(k,'c')
            CBF_ON = ~CBF_ON;
            fprintf('CBF safety filter: %s\n', ternary(CBF_ON,'ON','OFF'));
        end
    end

    function kr(~, evt)
        ud_kr = getappdata(fig,'keys');
        k  = lower(strrep(evt.Key,' ',''));
        if isfield(ud_kr,k), ud_kr.(k) = false; setappdata(fig,'keys',ud_kr); end
    end

    function on_close(src, ~)
        setappdata(src, 'quit', true);
        running = false;
        delete(src);
    end

    % ──────────────────────────────────────────────────────────────────────
    %  RE-ANCHORING  –  switch the reference frame from the old anchor A_old
    %  to A_new = camera pose at window slot 'idx' (the newest entry).
    %
    %  Frame relationships:
    %    A_new ≡ C_idx   →  R_{A_new}^{A_old} = win_RCjA(:,:,idx)
    %                        p_{A_new}^{A_old} = win_pCjA(:,idx)
    %
    %  For every window slot k the range vector transforms as:
    %    r_{A_new}^k = R_old2new * r_{A_old}^k
    %               = R_old2new * (rho_k * b_{f,k}^{A_old})
    %
    %  From this:
    %    rho_k_new = r_{A_new}^k (3)   = rho_k * (R_old2new(3,:) * b_old)
    %    b_{f,k}^{A_new} = r_{A_new}^k / rho_k_new          (b(3)=1 preserved)
    %    eta_k_new = ||b_{f,k}^{A_new}||
    %    R_{C_k}^{A_new} = R_old2new * R_{C_k}^{A_old}      (rotation product)
    %    p_{C_k}^{A_new} = R_old2new * (p_{C_k}^{A_old} - p_idx_A)
    % ──────────────────────────────────────────────────────────────────────
    function reanchor_to(idx)
        % Rotation and translation from old anchor to new anchor
        R_old2new = win_RCjA(:,:,idx)';   % R_{A_old}^{A_new}  (3×3, valid SO3)
        p_idx_A   = win_pCjA(:, idx);     % p_{C_idx}^{A_old}

        % ── 1. Update world-frame anchor state ───────────────────────────
        % R_{A_new}^W = R_{A_old}^W * R_{A_new}^{A_old}
        %             = R_A_W       * win_RCjA(:,:,idx)
        % p_{A_new}^W = R_A_W * p_{A_new}^{A_old} + p_{A_old}^W
        p_A_W = R_A_W * p_idx_A + p_A_W;
        R_A_W = R_A_W * win_RCjA(:,:,idx);

        % ── 2. Re-express feature in new anchor ──────────────────────────
        % p_f^{A_new} = R_old2new * (p_f^{A_old} - p_idx_A)
        p_f_A   = R_old2new * (p_f_A - p_idx_A);
        map_A.P = p_f_A;
        map_A.N = size(p_f_A, 2);

        % ── 3. Re-express every window entry in new anchor frame ─────────
        n_win = size(win_bf, 2);
        n_feat = size(win_bf, 3);
        for t = 1:n_win
            for f = 1:n_feat
                b_old = win_bf(:, t, f);              % bearing in A_old  (b(3)=1)
                if norm(b_old) < 1e-6, continue; end
                
                % Range vector in new anchor (direction only, scaled by rho)
                %   r_{A_new} = R_old2new * (rho_k * b_old)
                % We work with the normalised form then scale:
                r_new_dir = R_old2new * b_old;     % = r_{A_new} / rho_k
    
                % New depth: third component of the new range vector
                scale = r_new_dir(3);              % = rho_k_new / rho_k
    
                if abs(scale) < 1e-10
                    % Degenerate: feature lands on the new anchor's image plane.
                    % Keep old values to avoid division by zero; the slot will
                    % be evicted shortly (it is the oldest entry).
                    continue
                end
    
                % Update bearing (must satisfy b(3)=1 in A_new)
                win_bf(:, t, f)  = r_new_dir / scale;
    
                % Update depth
                win_rho(1, t, f)    = win_rho(1, t, f) * scale;
    
                % Update norm
                win_eta(1, t, f)    = norm(win_bf(:, t, f));
            end
            
            % Rotation: R_{C_k}^{A_new} = R_old2new * R_{C_k}^{A_old}
            win_RCjA(:,:,t) = R_old2new * win_RCjA(:,:,t);

            % Position: p_{C_k}^{A_new} = R_old2new*(p_{C_k}^{A_old} - p_idx_A)
            win_pCjA(:, t)  = R_old2new * (win_pCjA(:, t) - p_idx_A);
        end
    end

end % interactive_sim

%% ══════════════════════════ HELPER FUNCTIONS ══════════════════════════════

function d = n2d(p)
% NED → ENU-visual display (x=East, y=North, z=Up)
d = [p(2,:); p(1,:); -p(3,:)];
end

function s = false_keys()
s = struct('w',false,'s',false,'uparrow',false,'downarrow',false, ...
           'rightarrow',false,'leftarrow',false,'d',false,'a',false);
end

function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end
