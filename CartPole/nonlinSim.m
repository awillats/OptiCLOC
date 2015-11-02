function [x_k] = nonlinSim(x0, u_new, fnF, horizon)

global dt;
global nX;
global nU;

x_k = zeros(nX, horizon);
df = zeros(nX, horizon);
lenArgs = nX+nU;
nCells = ones(1,lenArgs);

x_k(:, 1) = x0;

for k = 2:horizon
        args = [x_k(:,k-1); u_new(:,k-1)];
        argsC = mat2cell(args', [1], nCells);
        df(:, k) = fnF(argsC{:});
        x_k(:, k) = x_k(:, k-1) + df(:,k)*dt;
end

end