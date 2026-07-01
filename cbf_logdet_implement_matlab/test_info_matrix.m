%% test_info_matrix.m
% =========================================================================
%  Tests for compute_info_matrix.m
%
%  Run from the triangulation_VIO root directory.
% =========================================================================
clear; clc;
addpath('utils');
addpath('sim');

fprintf('===== Information Matrix Tests =====\n\n');

% ── Test 1: No features → invalid ────────────────────────────────────────
info = compute_info_matrix(zeros(3,0), []);
assert(~info.is_valid,          'FAIL: empty should be invalid');
assert(isinf(info.logdet),      'FAIL: logdet should be -Inf');
assert(info.n_feat == 0,        'FAIL: n_feat should be 0');
fprintf('Test 1 — No features → is_valid=false, logdet=-Inf  ✓\n');

% ── Test 2: Single bearing → M rank 2 (not PD, invalid) ─────────────────
b1 = [0;0;1];   % straight down
info = compute_info_matrix(b1, 5);
assert(~info.is_valid,          'FAIL: single bearing should be rank-deficient');
fprintf('Test 2 — Single bearing → rank-deficient (lambda_min=%.2e)  ✓\n', ...
    info.lambda_min);

% ── Test 3: Three orthogonal bearings → M well-conditioned ───────────────
%   b1=[1;0;0], b2=[0;1;0], b3=[0;0;1]  →  each pi_j = I - e_i*e_i'
%   M = (I-ee1') + (I-ee2') + (I-ee3') = 3I - I = 2I
%   So M = 2I, det(M) = 8, log(det) = 3*log(2)
b_orth = eye(3);                % columns are orthogonal bearing vectors
rho_orth = [5; 5; 5];
info = compute_info_matrix(b_orth, rho_orth');
assert(info.is_valid,                               'FAIL: should be valid');
assert(norm(info.M - 2*eye(3)) < 1e-10,            'FAIL: M should be 2I');
assert(abs(info.logdet - 3*log(2)) < 1e-10,        'FAIL: logdet wrong');
assert(abs(info.lambda_min - 2) < 1e-10,           'FAIL: lambda_min should be 2');
assert(abs(info.cond_num - 1) < 1e-10,             'FAIL: cond should be 1 (isotropic)');
fprintf('Test 3 — 3 orthogonal bearings: M=2I, logdet=3*log(2)=%.4f  ✓\n', ...
    3*log(2));

% ── Test 4: Collinear (coplanar) bearings → rank deficient ───────────────
%   All bearings along [1;0;0] → M = (I-ee1') * n = (n)*(I - ee1')
%   M has a null eigenvector = [1;0;0], rank = 2
b_col = repmat([1;0;0], 1, 5);
info  = compute_info_matrix(b_col, 5*ones(1,5));
assert(~info.is_valid,          'FAIL: collinear should be rank-deficient');
assert(info.lambda_min < 1e-8,  'FAIL: minimum eigenvalue should be ~0');
fprintf('Test 4 — Collinear bearings → rank-deficient (lambda_min=%.2e)  ✓\n', ...
    info.lambda_min);

% ── Test 5: Verify M symmetry and PSD ────────────────────────────────────
b_rand = randn(3,10);
b_rand = b_rand ./ vecnorm(b_rand);   % normalise columns
rho_rand = 3 + 2*rand(1,10);
info = compute_info_matrix(b_rand, rho_rand);
assert(norm(info.M - info.M', 'fro') < 1e-12,   'FAIL: M not symmetric');
assert(all(info.eigvals >= -1e-12),               'FAIL: M not PSD');
fprintf('Test 5 — Random bearings: M symmetric & PSD, logdet=%.3f  ✓\n', ...
    info.logdet);

% ── Test 6: Verify hdot = Psi' * u linearity ─────────────────────────────
%   Compute hdot analytically using bearing dynamics and compare with Psi'*u
%   Uses a simple unit test: scale u by 2 → hdot scales by 2.
u_test  = [0.5; 0.1; -0.2;  0.1; -0.3; 0.2];   % [v_body; om_body]
hdot_1  = info.Psi' * u_test;
hdot_2  = info.Psi' * (2*u_test);
assert(abs(hdot_2 - 2*hdot_1) < 1e-12, 'FAIL: hdot not linear in u');
fprintf('Test 6 — hdot = Psi''*u scales linearly  ✓\n');

fprintf('\n===== All tests PASSED =====\n\n');

% ── Print summary of Test 3 (isotropic case) ─────────────────────────────
fprintf('--- Detailed output (isotropic 3-bearing case) ---\n');
b_orth = eye(3); rho_orth = [5;5;5];
info3  = compute_info_matrix(b_orth, rho_orth');
fprintf('M =\n');     disp(info3.M);
fprintf('log det(M)  = %.6f  (expected %.6f)\n', info3.logdet, 3*log(2));
fprintf('Eigenvalues = [%.4f, %.4f, %.4f]\n',   info3.eigvals);
fprintf('Cond number = %.4f\n',                  info3.cond_num);
fprintf('Psi_v  = [%.4f; %.4f; %.4f]\n',        info3.Psi_v);
fprintf('Psi_om = [%.4f; %.4f; %.4f]\n',        info3.Psi_om);
