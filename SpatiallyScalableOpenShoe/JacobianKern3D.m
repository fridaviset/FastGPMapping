function JacobianPhi=JacobianKern3D(x,xs,sigma_SE,l_SE)
rows=size(xs,2);
JacobianPhi=zeros(3,3,rows);
for i=1:rows
    nom=norm((x-xs(:,i))./l_SE')^2;
    den=2;
    core_alt=nom/den;
    dist=(x-xs(:,i))./l_SE;
    JacobianPhi(:,:,i)=sigma_SE^2./l_SE^2*exp(-core_alt)*(dist*dist'-eye(3));
end

end