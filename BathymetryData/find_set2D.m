function [set1, set2, xu_set, xu1_set, xu2_set]=find_set2D(x,Omega,l_u1,l_u2,N_u,x1_u,x2_u,r)
%Copyright (C) 2022 by Frida Viset

%This functions selects the set of inducing points that
%along each dimension is closer than r to the point x
ls1=max(floor((x(1)-r(1)-Omega(1,1))./l_u1+2),1);
us1=min(ceil((x(1)+r(1)-Omega(1,1))./l_u1),N_u(1));
set1=ls1:us1;
ls2=max(floor((x(2)-r(2)-Omega(2,1))./l_u2+2),1);
us2=min(ceil((x(2)+r(2)-Omega(2,1))./l_u2),N_u(2));
set2=ls2:us2;

%Find the corresponding x_u, in the correct order...
[xu2_set, xu1_set]=meshgrid(x2_u(set2),x1_u(set1));
xu_set=[xu1_set(:) xu2_set(:)]';
end
