function [set_all, xu_set, set1, set2, set3, xu1_set, xu2_set, xu3_set]=find_set3D(x,Omega,l_u1,l_u2,l_u3,N_u,x1_u,x2_u,x3_u,r)
%This functions selects the set of inducing points that
%along each dimension is closer than r to the point x

ls1=max(floor((x(1)-r-Omega(1,1))./l_u1+2),1);
us1=min(ceil((x(1)+r-Omega(1,1))./l_u1),N_u(1));
set1=ls1:us1;
ls2=max(floor((x(2)-r-Omega(2,1))./l_u2+2),1);
us2=min(ceil((x(2)+r-Omega(2,1))./l_u2),N_u(2));
set2=ls2:us2;
ls3=max(floor((x(3)-r-Omega(3,1))./l_u3+2),1);
us3=min(ceil((x(3)+r-Omega(3,1))./l_u3),N_u(3));
set3=ls3:us3;

%Find the corresponding x_u, in the correct order...
[xu2_set, xu1_set,xu3_set]=meshgrid(x2_u(set2),x1_u(set1),x3_u(set3));
xu_set=[xu1_set(:) xu2_set(:) xu3_set(:)]';
[set2_all, set1_all, set3_all]=meshgrid(set2,set1,set3);
set_all=set1_all(:)+(set2_all(:)-1)*N_u(1)+(set3_all(:)-1)*N_u(1)*N_u(2);
end