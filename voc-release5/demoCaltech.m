function demoCaltech()
init_env;

dataDir = '/media/Volume_1/capstone2/caltech_ped_dataset/data-USA/';
resultFile = 'results/caltechPed';
reapply = 0;
pLoad={'lbls',{'person'},'ilbls',{'people'},'squarify',{3,.41}};

bbsNm=[resultFile 'Dets.txt'];
if(reapply && exist(bbsNm,'file')), delete(bbsNm); end
if(reapply || ~exist(bbsNm,'file'))
    load('caltechped_final');
    
    % load('VOC2010/person_grammar_final');
    % model.class = 'person grammar';

    model = model;
    imgNms=bbGt('getFiles',{[dataDir 'test/images']});
    n=length(imgNms)
    bbs = cell(n,1);
    parfor imgIdx=1:n
        bbs{imgIdx} = test(imgNms{imgIdx}, model, -0.6);
        disp(imgIdx);
        %pause(0.1);
        %break;
    end
    for i=1:n
        if ~isempty(bbs{i})
            bbs{i}=[ones(size(bbs{i},1),1)*i bbs{i}]; 
            % convert voc5 dt bbs into piotr format
            bbs{i}(3) = bbs{i}(3) - bbs{i}(1);
            bbs{i}(4) = bbs{i}(4) - bbs{i}(2);
        else bbs{i}=ones(0,6); 
        end
    end
    bbs=cell2mat(bbs);
    d=fileparts(bbsNm); if(~isempty(d)&&~exist(d,'dir')), mkdir(d); end
    dlmwrite(bbsNm,bbs);
end

[gt,dt] = bbGt('loadAll',[dataDir 'test/annotations'],bbsNm,pLoad);
if(~exist('imgNms')), imgNms=bbGt('getFiles',{[dataDir 'test/images']}); end
imgIdx = 1;
imgIdx = length(imgNms);
while imgIdx<length(imgNms)
    I=imread(imgNms{imgIdx});
    fh = figure(1); 
    im(I); 
    bbApply('draw',dt{imgIdx},'r'); %pause(.1);
    bbApply('draw',gt{imgIdx});
    kkey = get(gcf,'CurrentCharacter');
    while isempty(kkey)
        pause(0.1);
        kkey = get(gcf,'CurrentCharacter');
    end
    if ~isempty(dt{imgIdx})
        disp(dt{imgIdx});
        pause(0.1);
    end
    if kkey == 29
        imgIdx = imgIdx + 1;
    elseif kkey == 28
        imgIdx = imgIdx - 1;
    end
end

% load('VOC2010/person_grammar_final');
% model.class = 'person grammar';
% %model.vis = @() visualize_person_grammar_model(model, 6);
% test('000061.jpg', model, -0.6);

%% test detector and plot roc (see acfTest)
[~,~,gt,dt]=acfTest('name',resultFile,'imgDir',[dataDir 'test/images'],...
  'gtDir',[dataDir 'test/annotations'],'pLoad',[pLoad, 'hRng',[50 inf],...
  'vRng',[.65 1],'xRng',[5 635],'yRng',[5 475]],'show',2,'reapply',0);

%% optionally show top false positives ('type' can be 'fp','fn','tp','dt')
if( 0 ), bbGt('cropRes',gt,dt,imgNms,'type','fn','n',50,...
    'show',3,'dims',opts.modelDs([2 1])); end


function bbox_res = test(imname, model, thresh)
% load and display image
im = imread(imname);
% im = imresize(im,0.9);
% clf;
% image(im);

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
