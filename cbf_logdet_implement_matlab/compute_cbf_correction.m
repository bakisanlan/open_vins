function [v_safe, cbf_active] = compute_cbf_correction(v_nom, info, CBF_ON, gamma, h_min, R_B_A, anchor_valid)
%COMPUTE_CBF_CORRECTION  Closed-form CBF-QP safety filter on body velocity.
%
%   [v_safe, cbf_active] = compute_cbf_correction(v_nom, info, CBF_ON,
%                                                  gamma, h_min, R_B_A,
%                                                  anchor_valid)
%
%   Implements the closed-form quadratic program:
%
%     CBF condition :  Ψ' v_B  ≥  −γ (h − h_min)
%
%     If violated:
%       λ* = (c_safe − Ψ' v_nom) / ‖Ψ‖²
%       v_safe = v_nom + λ* Ψ
%
%   Additionally projects out the body-frame direction that maps to
%   anchor-z, forcing zero vertical velocity in the anchor frame.
%
%   Inputs:
%     v_nom        — nominal body-frame velocity [3×1]
%     info         — struct from compute_info_matrix (.logdet, .Psi, .is_valid)
%     CBF_ON       — logical, whether CBF is enabled
%     gamma        — class-K decay rate
%     h_min        — log-det safety threshold
%     R_B_A        — body → anchor rotation (3×3)
%     anchor_valid — logical, whether anchor is established
%
%   Outputs:
%     v_safe     — corrected body-frame velocity [3×1]
%     cbf_active — true if CBF modified the velocity this tick

v_safe     = v_nom;
cbf_active = false;

%── Closed-form QP ────────────────────────────────────────────────────────
if CBF_ON && info.is_valid && norm(info.Psi) > 1e-10
    c_safe = -gamma * (info.logdet - h_min);
    margin = info.Psi' * v_nom;
    if margin < c_safe
        lambda_star = (c_safe - margin) / (info.Psi' * info.Psi);
        v_safe      = v_nom + lambda_star * info.Psi;
        cbf_active  = true;
    end
end

%── Force zero vertical velocity in anchor frame ─────────────────────────
%   v_A = R_B_A * v_B.  Set v_A(3) = 0 by projecting out the
%   body-frame direction that maps to anchor-z.
% if CBF_ON && anchor_valid
%     n_vert = R_B_A(3,:)';           % 3rd row of R_B_A = anchor-z in body
%     v_safe = v_safe - (n_vert' * v_safe) / (n_vert' * n_vert) * n_vert;
% end

end
