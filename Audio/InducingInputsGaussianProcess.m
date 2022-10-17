function [mu, var, x_u, train_time, inf_time]=InducingInputsGaussianProcess(x,y,x_s,uL,lL,sigma_y,N_u,sigma_SE,l_SE)
%Copyright (C) 2022 by Frida Viset

tic;
%Inducing input locations
x_u=linspace(lL,uL,N_u); 
%Pre-processing step
phi=sigma_SE^2.*exp(-(x'-x_u).^2./(2*l_SE^2));
I=phi'*phi;
iota=phi'*y';
train_time=toc;

%Inference step
N_s=size(x_s,2);
mu=zeros(N_s,1);
var=zeros(N_s,1);

tic;
phi=Kern(x_s(1),x_u,sigma_SE,l_SE);
Kuu=Kern(x_u,x_u,sigma_SE,l_SE);
temp=I+sigma_y^2.*Kuu;
temp=inv(temp);
mu(1)=phi*temp*iota;
var(1)=sigma_y^2*phi*temp*phi';
inf_time=toc;
Kuu_inv=inv(Kuu);

for i=2:N_s
    tic;
    phi=Kern(x_s(i),x_u,sigma_SE,l_SE);
    Kss=Kern(x_s(i),x_s(i),sigma_SE,l_SE);
    Qss=phi*Kuu_inv*phi';
    mu(i)=phi*temp*iota;
    var(i)=sigma_y^2*phi*temp*phi'+Kss-Qss;
end

end