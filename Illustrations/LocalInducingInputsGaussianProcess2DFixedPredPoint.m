function [mu, var, max_set_length, I]=LocalInducingInputsGaussianProcess2DFixedPredPoint(x,y,x_s,x_pred,Omega,N_u1,N_u2,sigma_SE,l_SE,r,sigma_y)
%Copyright (C) 2022 by Frida Viset

x1_u=linspace(Omega(1,1),Omega(1,2),N_u1); %Inducing inputs along first dimension
x2_u=linspace(Omega(2,1),Omega(2,2),N_u2); %Inducing inputs along second dimension
l_u1=x1_u(2)-x1_u(1); %Distance between two inducing inputs along dim 1
l_u2=x2_u(2)-x2_u(1); %Distance between two inducing inputs along dim 2

%Now, store and refer to bigger structures
I=zeros(N_u1,N_u2,N_u1,N_u2);
iota=zeros(N_u1,N_u2);

%Learning
N=size(x,2);
for t=1:N
    %Decide upon the set to use
    [set1, set2, xu_set]=find_set(x(:,t),Omega,l_u1,l_u2,N_u1,N_u2,x1_u,x2_u,r*2);

    %Perform the information matrix update
    phi=Kern(x(:,t),xu_set,sigma_SE,l_SE);
    Z=reshape(phi,length(set1),length(set2));
    I_mat=phi'*phi;
    iota_mat=phi'*y(t);
    I(set1,set2,set1,set2)=I(set1,set2,set1,set2)+reshape(I_mat,length(set1),length(set2),length(set1),length(set2));
    iota(set1,set2)=iota(set1,set2)+reshape(iota_mat,length(set1),length(set2));
end

%Prediction
N_s=size(x_s,2);
mu=zeros(N_s,1);
var=zeros(N_s,1);
max_set_length=0;
for t=1:N_s

%Calculate the set in a smarter way.
[set1, set2, xu_set]=find_set(x_pred,Omega,l_u1,l_u2,N_u1,N_u2,x1_u,x2_u,r);

M=length(set1)*length(set2);
I_mat_recon=reshape(I(set1,set2,set1,set2),M,M);
iota_mat_recon=reshape(iota(set1,set2),M,1);

phi=Kern(x_s(:,t),xu_set,sigma_SE,l_SE);
Kuu=Kern(xu_set,xu_set,sigma_SE,l_SE);
Kss=Kern(x_s(:,t),x_s(:,t),sigma_SE,l_SE);
Qss=phi*(Kuu\(phi'));
temp=I_mat_recon+sigma_y^2.*Kuu;
mu(t)=phi*(temp\iota_mat_recon);
var(t)=diag(sigma_y^2*phi*(temp\(phi')))+Kss-Qss;
max_set_length=max(max_set_length,M);

end


disp(['Max number of inducing inputs used: ',num2str(max_set_length)]);

end