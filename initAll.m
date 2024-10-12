clear;clc;close all;
load('parameterinit_initial.mat');

nl = linedata(:,1);
nr = linedata(:,2);
X  = linedata(:,4);


nbr=length(nl);
nbus=max(max(nl),max(nr));
Z=X;
y=ones(nbr,1)./Z;

Y=zeros(nbus,nbus);

for k=1:nbr
    if (nl(k)>0)&&(nr(k)>0)
        Y(nl(k),nr(k))=Y(nl(k),nr(k))-y(k);
        Y(nr(k),nl(k))=Y(nl(k),nr(k));
    end
end

for i=1:nbus
    for k=1:nbr
        if nl(k)==i||nr(k)==i
            Y(i,i)=Y(i,i)+y(k);
        end
    end
end
node_in = [30,31];
node_in_len = length(node_in);
node_out = setdiff(1:39,node_in);
Ll = Y(node_in,node_in)-Y(node_in,node_out)*(Y(node_out,node_out)\Y(node_out,node_in));
Vg   =opdata(:,4);
Theta=opdata(:,5)*pi/180;
for i = 1 : node_in_len
    for j = 1 : node_in_len
        if i~=j
            L(i,j) = Vg(i)*Vg(j)*Ll(i,j)*cos(Theta(i)-Theta(j));
        end
    end
end
for i = 1 : node_in_len
    L(i,i) = -sum(L(:,i));
end
ESinit.Kir(1:node_in_len) = 50*ones(node_in_len,1);
ESinit.Kdroop(1:node_in_len) = 100*ones(node_in_len,1);
m = ESinit.Kir(1:node_in_len);
d = ESinit.Kdroop(1:node_in_len);
save('parameterinit.mat');