clear all; close all; clc

N = 500;
x = zeros(2, N+1);
T = 85;

dt = 0.1;
om = 2*pi/(T*dt);
A_ct = [0, om; -om, 0];
har_ct = ss(A_ct, [], [] ,[]);
har_dt = c2d(har_ct, dt);
A = har_dt.A;

x(:, 1) = [0.6; 0];



for k = 1: N
    x(:, k+1) = A*x(:, k);
end

figure(1); hold on
plot([1: N+1], x(1, :));