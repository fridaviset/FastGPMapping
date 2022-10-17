function [mu, var, x_u, max_set_length, train_time, inf_time]=LocalInducingInputsGaussianProcess(x,y,x_s,uL,lL,sigma_y,N_u,sigma_SE,l_SE,r)
%Copyright (C) 2022 by Frida Viset

tic;

x_u=linspace(lL,uL,N_u); %Inducing input locations
l_u=x_u(2)-x_u(1); %Distance between two inducing points
N=size(x,2); %Number of measurements
I=zeros(N_u,N_u); %Pre-allocate space for the information matrix
iota=zeros(N_u,1); %Pre-allocate space for the information vector
max_set_length=0;

for t=1:N
    %Calculate the set as a function surrounding the current location
    ls1=max(floor((x(t)-2*r-lL)./l_u+2),1);
    us1=min(ceil((x(t)+2*r-lL)./l_u),N_u);
    set=ls1:us1;
    max_set_length=max(max_set_length,length(set));
    phi=sigma_SE^2.*exp(-(x(t)'-x_u(set)).^2./(2*l_SE^2));
    I(set,set)=I(set,set)+phi'*phi;
    iota(set)=iota(set)+phi'*y(t);
end
train_time=toc;

N_s=size(x_s,2);
mu=zeros(N_s,1);
var=zeros(N_s,1);
max_set_length=0;
tic;
for i=1:N_s
    tic;
    ls1=max(floor((x_s(i)-r-lL)./l_u+2),1);
    us1=min(ceil((x_s(i)+r-lL)./l_u),N_u);
    set=ls1:us1;
    max_set_length=max(max_set_length,length(set));
    phi=Kern(x_s(i),x_u(set),sigma_SE,l_SE);
    temp=I(set,set)+sigma_y^2.*Kern(x_u(set),x_u(set),sigma_SE,l_SE);
    temp=inv(temp);
    mu(i)=phi*temp*iota(set);
    var(i)=sigma_y^2*phi*temp*phi';
end
inf_time=toc./N_s;

end