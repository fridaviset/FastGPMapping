function [set1, set2, xu_set, xu1_set, xu2_set]=find_set(x,Omega,l_u1,l_u2,N_u1,N_u2,x1_u,x2_u,dist)
%Copyright (C) 2022 by Frida Viset

ls1=max(floor((x(1)-dist-Omega(1,1))./l_u1+2),1);
us1=min(ceil((x(1)+dist-Omega(1,1))./l_u1),N_u1);
set1=ls1:us1;
ls2=max(floor((x(2)-dist-Omega(2,1))./l_u2+2),1);
us2=min(ceil((x(2)+dist-Omega(2,1))./l_u2),N_u2);
set2=ls2:us2;
[xu1_set, xu2_set]=meshgrid(x1_u(set1),x2_u(set2));
xu1_set=xu1_set';
xu2_set=xu2_set';
xu_set=[xu1_set(:) xu2_set(:)]';
end