function  [L0,Lx,Lxx,Lu,Luu,Lux] = runningCost(x_k, u_k, R)

global nX;
global nU;
global dt;

L0 =    dt * u_k' * R * u_k;
Lx =    dt * zeros(nX,1);
Lxx =   dt * zeros(nX);
Lu =    dt * R * u_k;
Luu =   dt * R;
Lux =   dt * zeros(nU,nX);

end
