function demoCaltech()
startup;

dataDir = '/media/Volume_1/capstone2/caltech_ped_dataset/data-USA/';
resultFile = 'results/caltechPed.txt';

load('VOC2010/person_final');
model = model;
imgNms=bbGt('getFiles',{[dataDir 'test/images']});
n=length(imgNms)
bbs = cell(n,1);
parfor imgIdx=1:n
    bbs{imgIdx} = test(imgNms{imgIdx}, model, -0.3);
    disp(imgIdx);
    %pause(0.1);
end
for i=1:n
    if ~isempty(bbs{i})
        bbs{i}=[ones(size(bbs{i},1),1)*i bbs{i}]; 
    else
        bbs{i}=ones(0,6);
    end
end
bbs=cell2mat(bbs);
d=fileparts(resultFile); if(~isempty(d)&&~exist(d,'dir')), mkdir(d); end
dlmwrite(resultFile,bbs);

% load('VOC2010/person_grammar_final');
% model.class = 'person grammar';
% %model.vis = @() visualize_person_grammar_model(model, 6);
% test('000061.jpg', model, -0.6);

function bbox_res = test(imname, model, thresh)
% load and display image
im = imread(imname);
%clf;
%image(im);

% detect objects
[ds, bs] = imgdetect(im, model, thresh);
top = nms(ds, 0.5);
bbox_res = bs;
if ~isempty(bs)
    if model.type == model_types.Grammar
        bs = [ds(:,1:4) bs];
    end
    
    %clf;
    if model.type == model_types.MixStar
        % get bounding boxes
        bbox = bboxpred_get(model.bboxpred, ds, reduceboxes(model, bs));
        bbox = clipboxes(im, bbox);
        top = nms(bbox, 0.5);
        bbox_res = bbox(top,:);
        %showboxes(im, bbox_res);
    else
        bbox_res = reduceboxes(model, bs(top,:));
        %showboxes(im, bbox_res); 
    end
end
