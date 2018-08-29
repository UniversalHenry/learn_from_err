function [neg,pos,detail] = get_img_acc(neg,pos,detail,batch,res,labels)
l = reshape(res(end-1).x.*labels,size(labels,3),size(labels,4))>0;
detail = cat(1,detail,cat(1,batch,gather(l))');
l = sum(l,1)./size(l,1);
t = 0.8;  %threshold for accuracy
l = (l>=t);
tmppos = [1:numel(l)];
tmppos = tmppos.*l;
tmppos = tmppos(tmppos>0);
tmpneg = [1:numel(l)];
tmpneg = tmpneg.*(1-l);
tmpneg = tmpneg(tmpneg>0);
pos = cat(2,pos,gather(tmppos));
neg = cat(2,neg,gather(tmpneg));