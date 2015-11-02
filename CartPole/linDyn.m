function [Fx, Fu] = linDyn(f, x, u)

global nX;
global nU;

for i = 1:nX
    for j = 1:nX
        Fx(i, j) = gradient(f(i), x(j));
    end
end

for i = 1:nX
    for j = 1:nU
        Fu(i, j) = gradient(f(i), u(j));
    end
end
end