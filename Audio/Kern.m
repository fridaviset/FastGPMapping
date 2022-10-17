function K=Kern(x,y,sigma_SE,l_SE)
%Copyright (C) 2022 by Frida Viset

K=sigma_SE^2.*exp(-(x'-y).^2./(2*l_SE^2));
end