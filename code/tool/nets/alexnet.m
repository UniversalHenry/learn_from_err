function net = alexnet(labelNum,lossWeight,opts)
netname='alexnet';
output_num=labelNum;
alexnet=load('../nets/imagenet-caffe-alex.mat');

net.meta=alexnet.meta;
for i = 1 : numel(alexnet.layers)-1
    net.layers{i}=alexnet.layers{i};
end
for i = 13 : numel(alexnet.layers)-1
    net.layers{i}.alpha = 1;
    if strcmp(alexnet.layers{i}.type,'conv') % && ~strcmp(alexnet.layers{i}.name(1:2),'fc')
        net.layers{i}.type='our_conv';
        net.layers{i}.weights=alexnet.layers{i}.weights;
        ts=net.layers{i}.size;
        tmp_weights_added=alexnet.layers{i};
%         tmp_weights_added={init_weight(opts, ts(1),ts(2),ts(3),ts(4) , 'single'), ...
%                               ones(net.layers{i}.size(4), 1, 'single')*opts.initBias};
        net.layers{i}.weights = {cat(3,net.layers{i}.weights{1},tmp_weights_added{1}),...
                              cat(2,net.layers{i}.weights{2},tmp_weights_added{2})};
        net.layers{i}.addnum = 1;
    end
end
% for i = 1 : numel(our_prenet.net.layers)
%     addnum = 1;
%     net.layers{i}=our_prenet.net.layers{i};
%     net.layers{i}.alpha = 0.5;
% end
% net = add_dropout(net, opts, 'fc--6') ;
% net = add_blockfc(net, opts, '7', 1, 1, 256,  4096, 1, 0) ;
% net = add_dropout(net, opts, 'fc--7') ;
% net = add_blockfc(net, opts, '8', 1, 1, 4096,  output_num, 1, 0) ;
net.layers(end) = [] ;
net.meta.classes.name={'pos'};
net.meta.classes.description={'pos'};
net.meta.normalization.imageSize=net.meta.normalization.imageSize(1:3);
net.meta.netname=netname;
clear alexnet tmp_weights_added;
end
