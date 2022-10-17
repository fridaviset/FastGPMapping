function [mu, var, preprocessingtime, inferencetime]=InducingPointGP3D(x,y,x_s,Omega,N_u,sigma_SE,l_SE,sigma_y)
%Copyright (C) 2022 by Frida Viset

x1_u=linspace(Omega(1,1),Omega(1,2),N_u(1)); %Inducing inputs along first dimension
x2_u=linspace(Omega(2,1),Omega(2,2),N_u(2)); %Inducing inputs along second dimension
x3_u=linspace(Omega(3,1),Omega(3,2),N_u(3)); %Inducing inputs along third dimension

%Define and store the full set of inducing points
[xu2m, xu1m,xu3m]=meshgrid(x2_u,x1_u,x3_u);
xu=[xu1m(:) xu2m(:) xu3m(:)]';

tic;
%Learning

Available_memory=4*10^8;
N=size(x,2);
m=size(xu,2);
I=zeros(m);
iota=zeros(m,1);
%Find the maximum amount of measurements we can store at any time
N_max=floor(Available_memory./(m*2));
intervals=1:N_max:N;
for i=1:size(intervals,2)-1
    interval=intervals(i):intervals(i+1)-1;
    phi_i=Kern3D(x(:,interval),xu,sigma_SE,l_SE);
    I=I+phi_i'*phi_i;
    iota=iota+phi_i'*y(interval)';
end
interval=intervals(end):N;
phi_i=Kern3D(x(:,interval),xu,sigma_SE,l_SE);
I=I+phi_i'*phi_i;
iota=iota+phi_i'*y(interval)';

preprocessingtime=toc;

%Prediction
N_s=size(x_s,2);
mu=zeros(N_s,1);
var=zeros(N_s,1);

tic;
Kuu=Kern3D(xu,xu,sigma_SE,l_SE);
temp=I+sigma_y^2.*Kuu;
temp=inv(temp);

for t=1:N_s
    
    phi=Kern3D(x_s(:,t),xu,sigma_SE,l_SE);
    mu(t)=phi*temp*iota;
    Qss=phi*temp*phi';
    Kss=Kern3D(x_s(:,t),x_s(:,t),sigma_SE,l_SE);
    var(t)=diag(sigma_y^2*phi*temp*phi')+Kss-Qss;
    if t==1
        inferencetime=toc;
    end
    
end

end