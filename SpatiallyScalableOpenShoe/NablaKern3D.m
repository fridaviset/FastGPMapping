function NablaPhi=NablaKern3D(x,xs,sigma_SE,l_SE)
rows=size(xs,2);
NablaPhi=zeros(3,rows);
for i=1:rows
    nom=norm((x-xs(:,i))./l_SE')^2;
    den=2;
    core_alt=nom/den;
    NablaPhi(:,i)=-sigma_SE^2./l_SE^2*exp(-core_alt)*(x-xs(:,i));
end

end