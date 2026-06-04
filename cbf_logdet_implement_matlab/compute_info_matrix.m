function info = compute_info_matrix(b_f_win, rho_win, eta_win, R_CjA_win, R_c2b, n_vis)
%COMPUTE_INFO_MATRIX  Triangulation information matrix and CBF gradient.
%
%   info = compute_info_matrix(b_f_win, rho_win, eta_win, R_CjA_win, R_c2b, n_vis)
%
% =========================================================================
%  Background  (see cbf_logdet_triangulation.tex)
% =========================================================================
%
%   All bearings b_{f,j} are expressed in the anchor frame A.
%
%   Generalised projection matrix:
%
%       π_j = η_j² I − b_{f,j} b_{f,j}'
%
%   Triangulation information matrix  (Eq. M_def):
%
%       M = Σ_j π_j = Σ_j (η_j² I − b_{f,j} b_{f,j}')
%
%   CBF gradient vector in BODY frame  (Eq. Psi_v):
%
%       Ψ = 2 Σ_j  1/(ρ_j η_j²)  (R_B^C)^T (R_{C_j}^A)^T  π_j M⁻¹ b_{f,j}
%         = 2 Σ_j  1/(ρ_j η_j²)  R_c2b  (R_{C_j}^A)'  π_j M⁻¹ b_{f,j}
%
%   where  (R_B^C)^T = R_c2b  (camera→body, fixed).
%
%   The barrier time derivative is:
%
%       ḣ = Ψ' v_B
%
%   Each term in the sum uses a DIFFERENT R_{C_j}^A from the window
%   (no simplification / factoring of the rotation).
%
% =========================================================================
%  Inputs
% =========================================================================
%
%   b_f_win    [3 × n]      Bearing vectors b_{f,j} in anchor frame
%   rho_win    [1 × n]      Depths ρ_j = r_A(3), anchor-frame z
%   eta_win    [1 × n]      Norms η_j = ||b_{f,j}||
%   R_CjA_win  [3 × 3 × n]  R_{C_j}^A  camera-j → anchor rotation per obs
%   R_c2b      [3 × 3]      Camera → body rotation (fixed)
%
% =========================================================================
%  Output struct
% =========================================================================
%
%   info.M          [3×3]   Information matrix  M = Σ π_j
%   info.M_inv      [3×3]   M⁻¹  ([] if singular)
%   info.Psi        [3×1]   CBF gradient in BODY frame
%   info.logdet     scalar  log det(M)
%   info.eigvals    [3×1]   Eigenvalues of M (ascending)
%   info.lambda_min scalar  Smallest eigenvalue
%   info.cond_num   scalar  Condition number  λ_max / λ_min
%   info.n_feat     scalar  Number of observations used
%   info.is_valid   bool    true if M is positive definite
%

[~, w_win, n_feat] = size(b_f_win);
n = w_win * n_feat;
if n == 0, info = empty_info(); return; end

sum_logdet = 0;
sum_Psi    = zeros(3,1);
sum_M      = zeros(3,3);
valid_feat_count = 0;
min_eig    = Inf;

for f = 1:n_feat
    %% ── 1. Build M_f for this feature ───────────────────────────────────
    M_f = zeros(3,3);
    obs_count = 0;
    for t = 1:w_win
        bfj   = b_f_win(:, t, f);
        if norm(bfj) < 1e-6, continue; end
        
        eta_j = eta_win(1, t, f);
        M_f   = M_f + (eta_j^2 * eye(3) - bfj * bfj');
        obs_count = obs_count + 1;
    end
    
    if obs_count == 0, continue; end
    
    %% ── 2. Eigenvalue analysis for M_f ──────────────────────────────────
    eigvals    = sort(eig(M_f), 'ascend');
    lambda_min = eigvals(1);
    min_eig    = min(min_eig, lambda_min);
    
    if lambda_min > 1e-8
        %% ── 3. Compute logdet and Psi for valid feature ─────────────────
        M_inv = inv(M_f);
        logdet_f = sum(log(eigvals));
        
        Psi_f = zeros(3,1);
        for t = 1:w_win
            bfj   = b_f_win(:, t, f);
            if norm(bfj) < 1e-6, continue; end
            
            eta_j = eta_win(1, t, f);
            rho_j = rho_win(1, t, f);
            pi_j  = (eta_j^2 * eye(3) - bfj * bfj');
            Mb    = M_inv * bfj;                          % M_f⁻¹ b_{f,j}
            R_CjA = R_CjA_win(:,:,t);                    % R_{C_j}^A for this frame
    
            Psi_f = Psi_f + (1/(rho_j * eta_j^2)) * R_c2b * R_CjA' * (pi_j * Mb);
        end
        Psi_f = 2 * Psi_f;
        
        sum_logdet = sum_logdet + logdet_f;
        sum_Psi    = sum_Psi + Psi_f;
        sum_M      = sum_M + M_f;
        valid_feat_count = valid_feat_count + 1;
    end
end

if valid_feat_count == 0
    info = empty_info();
    return;
end

%% ── Pack averaged output ────────────────────────────────────────────────
info.M          = sum_M / valid_feat_count;
info.M_inv      = inv(info.M);
info.Psi        = sum_Psi / valid_feat_count;
info.logdet     = sum_logdet / valid_feat_count;
info.eigvals    = sort(eig(info.M), 'ascend');
info.lambda_min = min_eig;
info.cond_num   = max(info.eigvals) / max(min_eig, 1e-12);
info.n_feat     = valid_feat_count;
info.is_valid   = true;

end % compute_info_matrix

%% ── Helper: empty output when no observations ──────────────────────────
function info = empty_info()
    info.M          = zeros(3,3);
    info.M_inv      = [];
    info.Psi        = zeros(3,1);
    info.logdet     = -Inf;
    info.eigvals    = zeros(3,1);
    info.lambda_min = 0;
    info.cond_num   = Inf;
    info.n_feat     = 0;
    info.is_valid   = false;
end
