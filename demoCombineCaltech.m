function demoCombineCaltech()
init_env;
dataDir = '/media/Volume_1/capstone2/caltech_ped_dataset/data-USA/';
resultFile = 'results/CombineCaltechDets.txt';

%% load acf detector
t=load('models/AcfCaltechDetector');
detector = t.detector;
detector = acfModify(detector,'cascThr',-1,'cascCal',-.005);

%% load dpm model
t=load('voc-release5/VOC2010/person_final');
%t=load('VOC2010/person_grammar_final'); t.model.class = 'person grammar';
dpm_model = t.model;

%% run on a images
imgNms=bbGt('getFiles',{[dataDir 'test/images']});
n = length(imgNms);
bbs = cell(n,1);
bbsCombined = cell(n,1);
imgIdx = 740; %55, 333, 369, 530, 740      :900
while imgIdx<length(imgNms)
    I=imread(imgNms{imgIdx}); 
    fh = figure(1); im(I); 
    tic, bbs{imgIdx}=acfDetect(I,detector); toc
    bbApply('draw',bbs{imgIdx},'r');
    tic, bbsCombined{imgIdx}=postProcess(I,dpm_model,bbs{imgIdx}); toc
    bbApply('draw',bbsCombined{imgIdx},'g'); %pause(.1);
    kkey = get(gcf,'CurrentCharacter');
    while isempty(kkey)
        pause(0.1);
        kkey = get(gcf,'CurrentCharacter');
    end
    if ~isempty(bbs{imgIdx})
        %disp(bbs{imgIdx});
        %pause(0.1);
    end
    if kkey == 29, imgIdx = imgIdx + 1;
    elseif kkey == 28, imgIdx = imgIdx - 1;
    end
end
for i=1:n
    if ~isempty(bbs{i}), bbs{i}=[ones(size(bbs{i},1),1)*i bbs{i}]; 
    else bbs{i}=ones(0,6); end
end
bbs=cell2mat(bbs);
d=fileparts(resultFile); if(~isempty(d)&&~exist(d,'dir')), mkdir(d); end
dlmwrite(resultFile,bbs);
end

function bbs_res = postProcess(I,model,bbs)
if size(bbs,1) == 0
    bbs_res=bbs;
    return;
end
% apply threshold to bbs
% run dpm only on first n highest scored bbs
bbs_res=zeros(0,5);
for bbIdx=1:size(bbs,1)
    I_bb = I(max(1,bbs(bbIdx,2)):min(size(I,1),bbs(bbIdx,2)+bbs(bbIdx,4)),...
             max(1,bbs(bbIdx,1)):min(size(I,2),bbs(bbIdx,1)+bbs(bbIdx,3)),:);
    scale = 40/size(I_bb,2);
    I_bb = imresize(I_bb,scale);
    [ds, bs] = imgdetect(I_bb, model, -0.75);
    if ~isempty(bs)
        if model.type == model_types.Grammar
            bs = [ds(:,1:4) bs];
        end
        if model.type == model_types.MixStar
            % get bounding boxes
            bbox = bboxpred_get(model.bboxpred, ds, reduceboxes(model, bs));
            bbox = clipboxes(I, bbox);
            top = nms(bbox, 0.5);
            bbs_res(end+1,:) = bbox(top,:);
            %showboxes(im, bbox_res);
        else
            top = nms(ds, 0.5);
            bbs_res(end+1,:) = ds(top,[1,2,3,4,6]);
            %showboxes(im, bbox_res); 
        end
        bbs_res(end,1:4) = bbs_res(end,1:4)/scale;
        bbs_res(end,3)=bbs_res(end,3)-bbs_res(end,1);
        bbs_res(end,4)=bbs_res(end,4)-bbs_res(end,2);
        bbs_res(end,1)=bbs_res(end,1)+bbs(bbIdx,1);
        bbs_res(end,2)=bbs_res(end,2)+bbs(bbIdx,2);
    elseif ~isempty(ds)
        top = nms(ds, 0.5);
        bbs_res(end+1,:) = ds(top,:);
        bbs_res(end,3)=bbs_res(end,3)-bbs_res(end,1);
        bbs_res(end,4)=bbs_res(end,4)-bbs_res(end,2);
        bbs_res(end,1)=bbs_res(end,1)+bbs(bbIdx,1);
        bbs_res(end,2)=bbs_res(end,2)+bbs(bbIdx,2);
    else
        % this returns the original deteciton, if dpm fails
        %bbs_res=bbs;
    end
end
end