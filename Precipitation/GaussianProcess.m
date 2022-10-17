function [mu, var]=GaussianProcess(x,y,x_s,sigma_y,sigma_SE,l_SE)
N_s=size(x_s,2);
mu=zeros(N_s,1);
var=zeros(N_s,1);
K=Kern2D(x,x,sigma_SE,l_SE);
N=size(x,2);
temp_mat=inv(K+sigma_y^2.*eye(N));
temp=temp_mat*y';
for i=1:size(x_s,2)
Ks=Kern2D(x_s(:,i),x,sigma_SE,l_SE);
Kss=Kern2D(x_s(:,i),x_s(:,i),sigma_SE,l_SE);
mu(i)=Ks*temp;
var(i)=diag(Kss-Ks*temp_mat*Ks');
end
end


