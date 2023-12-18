clear; close all;

%Find the sparsity pattern
Indices_dyn=[];

%Inducing point locations
x1_u=-1.5:1:1.5;
x2_u=-1.5:1:1.5;
x3_u=-1:1:1;
r_thresh=1.9;
[xu2_set,xu1_set,xu3_set]=meshgrid(x2_u,x1_u,x3_u);
xu_set_full=[xu1_set(:) xu2_set(:) xu3_set(:)]';
N_u=size(xu_set_full,2);
N_u_2=size(x2_u,2)*size(x1_u,2);
N_u_1=size(x1_u,2);

Sparsity_pattern_dyn=zeros(N_u);
%figure; clf; surf(Sparsity_pattern,'Edgecolor','None');
for i=1:N_u
    for j=1:N_u
        if max(abs(xu_set_full(:,i)-xu_set_full(:,j)))<=r_thresh
            Indices_dyn=[Indices_dyn,[i; j]];
        else
        end
    end
end
linear_indices_dyn=sub2ind(size(Sparsity_pattern_dyn),Indices_dyn(1,:),Indices_dyn(2,:));
Sparsity_pattern_dyn(linear_indices_dyn)=1;
map = [[30, 77, 150]./255;
       [167, 200, 252]./255];
   box off; axis equal; axis off; colormap(map);
Sparsity_image=zeros(N_u,N_u,3);
Sparsity_image=permute(reshape(repmat(map(2,:),N_u,N_u),N_u,3,N_u),[1 3 2]);
for i=1:N_u
    for j=1:N_u
        if max(abs(xu_set_full(:,i)-xu_set_full(:,j)))<=r_thresh
            Sparsity_image(i,j,:)=map(1,:);
        end
    end
end
fac=10;
Sparsity_image_3D=zeros(fac*N_u,fac*N_u,3);
for rgb=1:3
    mask=ones(fac);
    mask(:,1)=0;
    mask(:,end)=0;
    mask(1,:)=0;
    mask(end,:)=0;
    Sparsity_image_3D(:,:,rgb)=kron(Sparsity_image(:,:,rgb),mask);
end
imwrite(Sparsity_image_3D,'Figures/Sparsity3D.png');

fac=fac*size(x3_u,2);
Sparsity_image_2D=zeros(fac*N_u_2,fac*N_u_2);
for rgb=1:3
    mask=ones(fac);
    mask(:,1)=0;
    mask(:,end)=0;
    mask(1,:)=0;
    mask(end,:)=0;
    Sparsity_image_2D(:,:,rgb)=kron(Sparsity_image(1:N_u_2,1:N_u_2,rgb),mask);
end
imwrite(Sparsity_image_2D,'Figures/Sparsity2D.png');

fac=fac*size(x2_u,2);
Sparsity_image_1D=zeros(fac*N_u_1,fac*N_u_1);
for rgb=1:3
     mask=ones(fac);
    mask(:,1)=0;
    mask(:,end)=0;
    mask(1,:)=0;
    mask(end,:)=0;
    Sparsity_image_1D(:,:,rgb)=kron(Sparsity_image(1:N_u_1,1:N_u_1,rgb),mask);
end
imwrite(Sparsity_image_1D,'Figures/Sparsity1D.png');