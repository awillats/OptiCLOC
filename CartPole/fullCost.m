function [cost] =  fullCost(x_k, u_new, p_k, phi_f, gamma, R)

global dt;

horizon = size(x_k, 2);

cost = 0;
for j =1:(horizon-1)
    
    cost = cost + gamma * u_new(:,j)' * R * u_new(:,j) * dt;
    
end

TerminalCost= (x_k(:,horizon) - p_k(:,horizon))' * phi_f * (x_k(:,horizon) - p_k(:,horizon));

cost = cost + TerminalCost;

end