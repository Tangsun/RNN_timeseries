clear all; close all; clc

N = 500;
beta_val = [0.8, 0.9, 1.0];
mu_val = [0.7, 0.8, 0.9];
T_val = [30, 60, 90];
x_0 = 0.6;
dt_sin = 0.1;

traj = zeros(N+1, 6);
traj(:, 1) = Henon_map_traj(N, beta_val(1), x_0);
traj(:, 2) = Henon_map_traj(N, beta_val(2), x_0);
traj(:, 3) = Henon_map_traj(N, beta_val(3), x_0);
traj(:, 4) = Ikeda_map_traj(N, mu_val(1), x_0);
traj(:, 5) = Ikeda_map_traj(N, mu_val(2), x_0);
traj(:, 6) = Ikeda_map_traj(N, mu_val(3), x_0);
traj(:, 7) = harmonic_lin(N, T_val(1), dt_sin, x_0);
traj(:, 8) = harmonic_lin(N, T_val(2), dt_sin, x_0);
traj(:, 9) = harmonic_lin(N, T_val(3), dt_sin, x_0);

all_traj = [];
cell_traj = cell(9, 1);
for i = 1: 9
    traj_i = [i*ones(N+1, 1), [0: 1: N]', traj(:, i)];
    all_traj = [all_traj; traj_i];
    cell_traj{i} = traj(:, i);
end

% figure(1); hold on
% plot([1: N+1], traj(:, 7), 'Color', 'k');
% plot([1: N+1], traj(:, 8), 'Color', 'b');
% plot([1: N+1], traj(:, 9), 'Color', 'r');

traj = all_traj;
tracks = cell_traj;
save('TrainSet','traj','tracks');

%% Interpolation and Extrapolation Test Set
N_var = 500;
beta_var = [0.75, 0.85, 0.95];
mu_var = [0.65, 0.75, 0.85];
T_var = [20, 50, 80];
% T_var = [20, 50, 100];
x_0 = 0.6;

var_traj = zeros(N_var+1, 9);
for i = 1: 3
    var_traj(:, i) = Henon_map_traj(N_var, beta_var(i), x_0);
    var_traj(:, 3+i) = Ikeda_map_traj(N_var, mu_var(i), x_0);
    var_traj(:, 6+i) = harmonic_lin(N_var, T_var(i), dt_sin, x_0);
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
save('TestSet_var','traj','tracks');


%% Switching Test Set
N_1 = 500; N_2 = 500; N_3 = 500;
N_sw = N_1 + N_2 + N_3;

sw_traj = zeros(N_sw, 3);
% switch 1: Training Data
beta_sw = 0.9; mu_sw = 0.8; T_sw = 60;
sw_seq_1 = Henon_map_traj(N_1, beta_sw, x_0);
sw_seq_2 = Ikeda_map_traj(N_2, mu_sw, sw_seq_1(end));
sw_seq_3 = harmonic_lin(N_3, T_sw, dt_sin, sw_seq_2(end));
sw_traj(:, 1) = [sw_seq_1(1: N_1); sw_seq_2(1: N_2); sw_seq_3(1: N_3)];

% switch 2: Henon to Ikeda to T (Interpolation)
beta_sw = 0.85; mu_sw = 0.75; T_sw = 80;
sw_seq_1 = Henon_map_traj(N_1, beta_sw, x_0);
sw_seq_2 = Ikeda_map_traj(N_2, mu_sw, sw_seq_1(end));
sw_seq_3 = harmonic_lin(N_3, T_sw, dt_sin, sw_seq_2(end));
sw_traj(:, 2) = [sw_seq_1(1: N_1); sw_seq_2(1: N_2); sw_seq_3(1: N_3)];

% switch 3: Henon to T to Ikeda (Exterpolation)
beta_sw = 0.75; mu_sw = 0.65; T_sw = 20;
sw_seq_1 = Henon_map_traj(N_1, beta_sw, x_0);
sw_seq_2 = Ikeda_map_traj(N_2, mu_sw, sw_seq_1(end));
sw_seq_3 = harmonic_lin(N_3, T_sw, dt_sin, sw_seq_2(end));
sw_traj(:, 3) = [sw_seq_1(1: N_1); sw_seq_2(1: N_2); sw_seq_3(1: N_3)];


all_sw = [];
cell_sw = cell(3, 1);
for i = 1: 3
    traj_i = [i*ones(N_sw, 1), [0: 1: N_sw-1]', sw_traj(:, i)];
    all_sw = [all_sw; traj_i];
    cell_sw{i} = sw_traj(:, i);
end

traj = all_sw;
tracks = cell_sw;
save('TestSet_sw','traj','tracks');

% figure(1);
% plot([1: N_sw], sw_traj(:, 1));

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

function xhar = harmonic_lin(N, T, dt, x_0)
    xs = zeros(2, N+1);
    om = 2*pi/(T*dt);
    A_ct = [0, om; -om, 0];
    har_ct = ss(A_ct, [], [] ,[]);
    har_dt = c2d(har_ct, dt);
    A = har_dt.A;
    
    xs(:, 1) = [x_0; 0];
    for k = 1: N
        xs(:, k+1) = A*xs(:, k);
    end

    xhar = xs(1, :)';
end

function xs = harmonic_sin(N, T, x_0, tv_flag)
    xs = zeros(N+1, 1);
    xs(1, 1) = x_0;
    for k = 1: N
        if tv_flag == 0
            xs(k+1) = xs(k) + sin(2*pi*xs(k)/T);
        else 
            xs(k+1) = xs(k) + sin(2*pi*k/T);
        end
    end
end