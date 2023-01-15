clear all; close all; clc

N = 500;
beta_val = [0.8, 0.9, 1.0];
mu_val = [0.7, 0.8, 0.9];
T_val = [20, 50, 80];
x_0 = 0.6;

traj = zeros(N+1, 6);
traj(:, 1) = Henon_map_traj(N, beta_val(1), x_0);
traj(:, 2) = Henon_map_traj(N, beta_val(2), x_0);
traj(:, 3) = Henon_map_traj(N, beta_val(3), x_0);
traj(:, 4) = Ikeda_map_traj(N, mu_val(1), x_0);
traj(:, 5) = Ikeda_map_traj(N, mu_val(2), x_0);
traj(:, 6) = Ikeda_map_traj(N, mu_val(3), x_0);
traj(:, 7) = sin_traj(N, T_val(1));
traj(:, 8) = sin_traj(N, T_val(2));
traj(:, 9) = sin_traj(N, T_val(3));

all_traj = [];
cell_traj = cell(9, 1);
for i = 1: 9
    traj_i = [i*ones(N+1, 1), [0: 1: N]', traj(:, i)];
    all_traj = [all_traj; traj_i];
    cell_traj{i} = traj(:, i);
end

% figure(1);
% plot([1: N+1], traj(:, 6));

traj = all_traj;
tracks = cell_traj;
save('TrainSet','traj','tracks');

%% Interpolation and Extrapolation Test Set
N_var = 500;
beta_var = [0.75, 0.85, 0.95];
mu_var = [0.65, 0.75, 0.85];
T_var = [15, 35, 65];
x_0 = 0.6;

var_traj = zeros(N_var+1, 9);
for i = 1: 3
    var_traj(:, i) = Henon_map_traj(N_var, beta_var(i), x_0);
    var_traj(:, 3+i) = Ikeda_map_traj(N_var, mu_var(i), x_0);
    var_traj(:, 6+i) = sin_traj(N_var, T_var(i));
end
all_var = [];
cell_var = cell(9, 1);
for i = 1: 9
    traj_i = [i*ones(N_var+1, 1), [0: 1: N_var]', var_traj(:, i)];
    all_var = [all_var; traj_i];
    cell_var{i} = var_traj(:, i);
end

traj = all_var;
tracks = cell_var;
save('TestSet_var', 'traj', 'tracks');

%% Switching Test Set
N_1 = 500; N_2 = 500; N_3 = 500;
N_sw = N_1 + N_2 + N_3;
beta_sw = 0.75; mu_sw = 0.65; T_sw = 15;

sw_traj = zeros(N_sw+1, 2);
% switch 1: Henon to Ikeda to T
sw_seq_1 = Henon_map_traj(N_1, beta_sw, x_0);
sw_seq_2 = Ikeda_map_traj(N_2, mu_sw, x_0);
sw_seq_3 = sin_traj(N_3, T_sw);
sw_traj(:, 1) = [sw_seq_1(1: N_1); sw_seq_2(1: N_2); sw_seq_3];

beta_sw = 0.85; mu_sw = 0.75; T_sw = 35;
% switch 2: Henon to T to Ikeda
sw_seq_1 = Henon_map_traj(N_1, beta_sw, x_0);
sw_seq_2 = sin_traj(N_2, T_sw);
sw_seq_3 = Ikeda_map_traj(N_3, mu_sw, x_0);
sw_traj(:, 2) = [sw_seq_1(1: N_1); sw_seq_2(1: N_2); sw_seq_3];


all_sw = [];
cell_sw = cell(2, 1);
for i = 1: 2
    traj_i = [i*ones(N_sw+1, 1), [0: 1: N_sw]', sw_traj(:, i)];
    all_sw = [all_sw; traj_i];
    cell_sw{i} = sw_traj(:, i);
end

traj = all_sw;
tracks = cell_sw;
save('TestSet_sw', 'traj', 'tracks');

% figure(1);
% plot([1: N_sw+1], sw_traj(:, 3));

%% Supporting functions
function xh_1 = Henon_map_traj(N, beta, x_0)
    xh_1 = zeros(N+1, 1);
    xh_2 = zeros(N+1, 1);
    xh_1(1) = x_0;
    for k = 1: N
        xh_1(k+1) = beta - 1.4*xh_1(k)^2 + xh_2(k);
        xh_2(k+1) = 0.3*xh_1(k);
    end
end

function xi_1 = Ikeda_map_traj(N, mu, x_0)
    xi_1 = zeros(N+1, 1);
    xi_2 = zeros(N+1, 1);
    m = zeros(N);
    xi_1(1) = x_0;
    for k = 1: N
        m(k) = 0.4 - 6/(1 + xi_1(k)^2 + xi_2(k)^2);
        xi_1(k+1) = 1 + mu*(xi_1(k)*cos(m(k)) - xi_2(k)*sin(m(k)));
        xi_2(k+1) = mu*(xi_1(k)*sin(m(k)) + xi_2(k)*cos(m(k)));
    end
end

function xs = sin_traj(N, T)
    xs = zeros(N+1, 1);
    for k = 1: N+1
        xs(k) = sin(2*pi*k/T);
    end
end