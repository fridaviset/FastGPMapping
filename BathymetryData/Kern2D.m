function K=Kern2D(x,y,sigma_SE,l_SE)
%Copyright (C) 2022 by Frida Viset

rows=size(x,2);
columns=size(y,2);
K=zeros(rows,columns);

for i=1:rows
    for j=1:columns
        nom=norm((x(:,i)-y(:,j))./l_SE')^2;
        den=2;
        core_alt=nom/den;
        K(i,j)=sigma_SE^2*exp(-core_alt);
    end
end

end