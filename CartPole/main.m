% Inverted pendulum DDP
clear; close all; clc;

%% Declare global vars
global dt;
global nX;
global nU;

%% Parameterization & Setup

% Discretization
dt = 0.01; %s

% Gravitational acc
g = 9.8; %m/s2
% mass in Kgr
m_c = 1.0; % cart
m_p = 0.01; % pole

% length meters
l = 0.25;

% Horizon 
tHorizon = 3; %s
horizon = round(tHorizon/dt);
time = (0:horizon-1) .* dt;

% Number of Iterations
n_iter = 100;

% Number of states
nX = 4;
% Number of control inputs
nU = 1;

% Terminal Cost:
phi_f = zeros(nX, 1);
phi_f(1) = 10;
phi_f(2) = 100;
phi_f(3) = 100;
phi_f(4) = 100;
phi_f = diag(phi_f, 0);
% Running action cost:
R = 0.1 * eye(nU);

% Initialize state matrix
phi = zeros(nX, nX, horizon);
% Initialize action matrix
beta = zeros(nX, nU, horizon);

% Initial Configuration:
x0 = zeros(nX,1);
% Initial Control:
u_k = zeros(nU,horizon-1);
du_k = zeros(nU,horizon-1);
% Initial trajectory:
x_k = zeros(nX,horizon);

% Target:
p_k = zeros(nX, horizon);
p_k(3,:) = pi;

% Learning Rate:
gamma = 0.2;
%% System Dynamics
syms t u pos posDot th thDot

f = sym('blank', [nX, 1]);
x = sym('blank', [nX, 1]);

x(1) = pos;
x(2) = posDot;
x(3) = th;
x(4) = thDot;

% Nonlinear dynamics
f(1) = x(2);
f(2) = (m_c + m_p*sin(x(3))^2)^(-1) * (u + m_p*sin(x(3))*(l*x(4)^2 + g*cos(x(3))));
f(3) = x(4);
f(4) = (l*(m_c+m_p*sin(x(3))^2))^(-1) * ...
    (-u*cos(x(3)) - m_p*l*x(4)^2*cos(x(3))*sin(x(3)) - (m_c+m_p)*g*sin(x(3)));

subVars = [x(:); u(:)];
fnF = matlabFunction(f, 'Vars', subVars); %nonlinear dynamics

%---------------------> Linearization of the dynamics
[Fx, Fu] = linDyn(f, x, u);

fnFx = matlabFunction(Fx, 'Vars', subVars);
fnFu = matlabFunction(Fu, 'Vars', subVars);

%%

for k = 1:n_iter

% Forward Propagate
for  j = 1:(horizon-1)

    % Quadratic running cost
    [L0(:, :, j), Lx(:, :, j), Lxx(:, :, j), Lu(:, :, j), Luu(:, :, j), Lux(:, :, j)]...
        = runningCost(x_k(:,j), u_k(:,j), R);
    
    % Get linearized dynamics.
    args = [x_k(:,j); u_k(:,j)];
    argsC = mat2cell(args', [1], ones(1, length(args)));
    
    dfx = fnFx(argsC{:});
    dfu = fnFu(argsC{:});
    
    phi(:,:,j) = eye(nX) + dfx * dt;
    beta(:,:,j) = dfu * dt;
    
end

% Backward Propagate
% Initialize backward pass
Vxx(:,:,horizon)= phi_f;
Vx(:,horizon) = phi_f * (x_k(:,horizon) - p_k(:,horizon)); 
V(horizon) = gamma * (x_k(:,horizon) - p_k(:,horizon))' * phi_f * (x_k(:,horizon) - p_k(:,horizon)); 

% back dat thang up.
for j = (horizon-1):-1:1
    
    Q0(:,:,j) = L0(:,:,j) + V(j+1);
    Qx(:,:,j) = Lx(:,:,j) + phi(:,:,j)' * Vx(:,j+1);
    Qu(:,:,j) = Lu(:,:,j) + beta(:,:,j)' * Vx(:,j+1);
    Qxx(:,:,j) = Lxx(:,:,j) + phi(:,:,j)' * Vxx(:,:,j+1) * phi(:,:,j);
    Quu(:,:,j) = Luu(:,:,j) + beta(:,:,j)' * Vxx(:,:,j+1) * beta(:,:,j);
    Qux(:,:,j) = Lux(:,:,j) + beta(:,:,j)' * Vxx(:,:,j+1) * phi(:,:,j);
    Qxu(:,:,j) = Qux(:,:,j)';
    
    invQuu = inv(Quu(:,:,j));
    kappa_k(:,:,j) = -invQuu * Qux(:,:,j);
    nu_k(:,j) = -invQuu * Qu(:,:,j);
        
    Vxx(:,:,j) = Qxx(:,:,j) + kappa_k(:,:,j)' * Qux(:,:,j) + ...
        Qxu(:,:,j)*kappa_k(:,:,j) + kappa_k(:,:,j)' * Quu(:,:,j) * kappa_k(:,:,j);
    
    Vx(:,j)= Qx(:,:,j) + kappa_k(:,:,j)' * Qu(:,:,j) + ...
        Qxu(:,:,j)*nu_k(:,j) + kappa_k(:,:,j)' * Quu(:,:,j) * nu_k(:,j);
    
    V(:,j) = Q0(:,:,j) + nu_k(:,j)' * Qu(:,:,j) + ...
        1/2 * nu_k(:,j)' * Quu(:,:,j) * nu_k(:,j);
    
end
%--------------------> Find the controls
dx = zeros(nX,1);
for i=1:(horizon-1)    
   du = nu_k(:,i) + kappa_k(:,:,i) * dx;
   dx = phi(:,:,i) * dx + beta(:,:,i) * du;  
   u_new(:,i) = u_k(:,i) + gamma * du;
end
u_k = u_new;
%--------------------> Simulation of the Nonlinear System
[x_k] = nonlinSim(x0, u_new, fnF, horizon);

[cost(:,k)] =  fullCost(x_k, u_new, p_k, phi_f, gamma, R);
xInit(k,:) = x_k(1,:);

disp(['Iteration ' num2str(k) ' cost:  ' num2str(cost(k)) ' .'])
end
%%
figure(1)
subplot(2, 2, 1)
plot(time, x_k(1,:), 'color', 'k', 'linewidth', 3); hold on
plot(time, p_k(1,:), 'color', 'g', 'linewidth', 3);
title('Position(t)')
xlabel('time  [s]')
ylabel('Position  [m]')
legend({'State'; 'Target'})
grid

subplot(2, 2, 2)
plot(time, x_k(2,:), 'color', 'k', 'linewidth', 3); hold on
plot(time, p_k(2,:), 'color', 'g', 'linewidth', 3);
title('Velocity(t)')
xlabel('time  [s]')
ylabel('Xdot  [rad/s]')
legend({'State'; 'Target'})
grid

subplot(2, 2, 3)
plot(time, x_k(3,:), 'color', 'k', 'linewidth', 3); hold on
plot(time, p_k(3,:), 'color', 'g', 'linewidth', 3);
title('Theta(t)')
xlabel('time  [s]')
ylabel('Theta  [rad]')
legend({'State'; 'Target'})
grid

subplot(2, 2, 4)
plot(time, x_k(4,:), 'color', 'k', 'linewidth', 3); hold on
plot(time, p_k(4,:), 'color', 'g', 'linewidth', 3);
title('ThetaDot(t)')
xlabel('time  [s]')
ylabel('ThetaDot  [rad/s]')
legend({'State'; 'Target'})
grid

figure(2)
subplot(2, 1, 1)
plot(time(1:length(u_k)), u_k, 'color', 'b', 'linewidth', 3);
title('Control')
xlabel('time  [s]')
ylabel('force  [N]')
grid

subplot(2, 1, 2)
plot(1:n_iter, log(cost), 'color', 'r', 'linewidth', 3)
title('Cost')
xlabel('No. Iteration')
ylabel('log(cost)')
grid