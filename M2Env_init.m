function [] = M2Env_init(flag1,flag2,flag3)
load(fullfile(pwd,'parameterinit.mat'));
type = fix(flag1*1000);
amp = fix(flag2*1000);
comdisconnect = fix(flag3*(node_in_len*2+1));% num*2条通讯联络线，一共num*2+1种工况
rng(type);
loadinitial= rand([2 1]);
loadnode = fix(loadinitial(1)*2) + 30; %总共个节点
loadsize = (loadinitial(2) - 0.5)*40;
dis = zeros(node_in_len,1);
if ismember(loadnode,node_in)
    dis(loadnode==node_in,1) = loadsize;
else
    loadn = zeros(length(node_out),1);
    loadn(loadnode==node_out,1) = loadsize;
    dis = Y(node_in,node_out)/(Y(node_out,node_out))*loadn;
end
rng(amp);
DataDelay = rand([node_in_len 2])*0.0;
commun = ones(node_in_len,2);
% if comdisconnect~=node_in_len*2
%     comdisconnectT = mod(comdisconnect,8)+1;
%     comdisconnectR = fix(comdisconnect/8)+1;
%     commun(comdisconnectT,comdisconnectR) = 0;
% end
save('datainit');

