function demoDPMCaltech()
init_env;
dataDir = '/media/Volume_1/capstone2/caltech_ped_dataset/data-USA/';
%dataDir = '/tmp/data-USA/';
resultFile = 'results/DPMCaltech';
gtDir = [dataDir 'test/annotations'];
dpmThresh = -0.9; %def: -0.9
hMin = 55; %def: 55
pLoad = [{'lbls',{'person'},'ilbls',{'people'},'squarify',{3,.41}},...
  'hRng',[hMin inf],'vRng',[.95 1],'xRng',[5 635],'yRng',[5 475]];

%% load dpm model
t=load('voc-release5/caltechped_final');
%t=load('voc-release5/person_segDPM_final');
%t=load('voc-release5/VOC2010/person_final');
%t=load('VOC2010/person_grammar_final'); t.model.class = 'person grammar';
dpm_model = t.model;

%% run on a images
bbsNm=[resultFile 'Dets.txt'];
imgNms=bbGt('getFiles',{[dataDir 'test/images']});
[gt,~] = bbGt('loadAll',gtDir,[],pLoad);
n = length(imgNms);
bbs = cell(n,1);
dts = zeros(n,1);
%load n1; load p1; load p2;

imgIdx = 1;
while imgIdx<length(imgNms)
    I=imread(imgNms{imgIdx});
    tic;
    bbs{imgIdx} = test(I,dpm_model,dpmThresh);
    dts(imgIdx) = toc;
    imgIdx = imgIdx + 1;
    disp(imgIdx);
end

for i=1:n
    if ~isempty(bbs{i}), bbs{i}=[ones(size(bbs{i},1),1)*i bbs{i}]; 
    else bbs{i}=ones(0,6); end
end
bbs=cell2mat(bbs);
d=fileparts(bbsNm); if(~isempty(d)&&~exist(d,'dir')), mkdir(d); end
dlmwrite(bbsNm,bbs);

hist(dts,100);
end

function bbox_res = test(im, model, thresh)
% detect objects
[ds, bs] = imgdetect(im, model, thresh);
top = nms(ds, 0.5);
bbox_res = ds(top,:);
if ~isempty(bs)
    if model.type == model_types.Grammar
        bs = [ds(:,1:4) bs];
    end
    
    %clf;
    if model.type == model_types.MixStar
        % get bounding boxes
        if isfield(model, 'bboxpred')
            bbox = bboxpred_get(model.bboxpred, ds, reduceboxes(model, bs));
        else
            bbox = ds(:,[1:4,6]);
        end
        bbox = clipboxes(im, bbox);
        top = nms(bbox, 0.5);
        try
            bbox_res = bbox(top(1),:);
        catch
            disp('ERR');
        end
        %showboxes(im, bbox_res);
    else
        bbox_res = reduceboxes(model, ds(top,:));
        %showboxes(im, bbox_res); 
    end
end
end