function demoCombineCaltech()
init_env;
dataDir = '/media/Volume_1/capstone2/caltech_ped_dataset/data-USA/';
resultFile = 'results/CombineCaltech';
gtDir = [dataDir 'test/annotations'];
silent = 1;
reapply = 0;
rewrite = 0;
time = 0;
showBBTest = 0;
showAlertTest = 1;

optForAlert = 1;
horizon = 220; %def 220
igDpmThresh = 23; %def: 23
hMinThresh = 1;

dpmThresh = -0.9; %def: -0.9
icfThresh = -0.004; %def: -0.004
hMin = 55; %def: 55
pLoad = [{'lbls',{'person'},'ilbls',{'people'},'squarify',{3,.41}},...
  'hRng',[hMin inf],'vRng',[.95 1],'xRng',[5 635],'yRng',[5 475]];

bbsAfter = zeros(0,4);

%% load acf detector
t=load('models/AcfCaltechDetector_depth3_8192');
detector = t.detector;
detector = acfModify(detector,'cascThr',-1,'cascCal',icfThresh);

%% load dpm model
t=load('voc-release5/caltechped_final');
%t=load('voc-release5/person_segDPM_final');
%t=load('voc-release5/VOC2010/person_final');
%t=load('VOC2010/person_grammar_final'); t.model.class = 'person grammar';
dpm_model = t.model;

%% run on a images
[gt,~] = bbGt('loadAll',gtDir,[],pLoad);
bbsNm=[resultFile 'Dets.txt'];
imgNms=bbGt('getFiles',{[dataDir 'test/images']});
if(reapply && rewrite && exist(bbsNm,'file')), delete(bbsNm); end
if(reapply || ~exist(bbsNm,'file'))
    n = length(imgNms);
    bbs = cell(n,1);
    bbsCombined = cell(n,1);
    
    %load n1; load p1; load p2;
    
    imgIdx = 1; %55, 333, 369, 530, 740, 7130
    if silent, imgIdx = 1; end
    while imgIdx<length(imgNms)
        I=imread(imgNms{imgIdx});
        %I=imread(imgNms{p2(imgIdx,1)});
        if ~silent, fh = figure(1); im(I); bbApply('draw',gt{imgIdx}(gt{imgIdx}(:,5)==0,:),'b'); end
        if time, tic; end
        bbs{imgIdx}=acfDetect(I,detector);
        %bbs{imgIdx}=bbs{imgIdx}(bbs{imgIdx}(:,4)>hMin,:);
        bbsAfter(end+1,1)=size(bbs{imgIdx},1);
        if time, toc; end
        if ~silent, bbApply('draw',bbs{imgIdx},'r'); end
        if time, tic, end
        [bbsCombined{imgIdx},bbsAfter(end,2),bbsAfter(end,3),bbsAfter(end,4)]=...
                            postProcess(I,dpm_model,bbs{imgIdx},...
                            igDpmThresh,dpmThresh,hMin,hMinThresh,horizon,optForAlert);
        if time, toc; end
        if ~silent, bbApply('draw',bbsCombined{imgIdx},'g'); end %pause(.1);
        if ~silent
            kkey = get(gcf,'CurrentCharacter');
            while isempty(kkey)
                pause(0.1);
                kkey = get(gcf,'CurrentCharacter');
            end
            if ~isempty(bbs{imgIdx})
                %disp(bbs{imgIdx});
                %pause(0.1);
            end
            if kkey == 29, imgIdx = imgIdx + 1
            elseif kkey == 28, imgIdx = imgIdx - 1
            end
        else
            imgIdx = imgIdx + 1;
            disp(imgIdx);
        end
    end
    
    for i=1:n
        if ~isempty(bbs{i}), bbs{i}=[ones(size(bbs{i},1),1)*i bbs{i}]; 
        else bbs{i}=ones(0,6); end
    end
    if rewrite || ~exist(bbsNm,'file')
        bbs=cell2mat(bbs);
        d=fileparts(bbsNm); if(~isempty(d)&&~exist(d,'dir')), mkdir(d); end
        dlmwrite(bbsNm,bbs);
    end
    
    bbs = bbsCombined;
    bbsNm=[resultFile 'Comb' 'Dets.txt'];
    if(exist(bbsNm,'file')), delete(bbsNm); end
    for i=1:n
        if ~isempty(bbs{i}), bbs{i}=[ones(size(bbs{i},1),1)*i bbs{i}]; 
        else bbs{i}=ones(0,6); end
    end
    if rewrite || ~exist(bbsNm,'file')
        bbs=cell2mat(bbs);
        d=fileparts(bbsNm); if(~isempty(d)&&~exist(d,'dir')), mkdir(d); end
        dlmwrite(bbsNm,bbs);
    end
end

bbsAfterSum=sum(bbsAfter)

%% test detector and plot roc (see acfTest)
if showBBTest && ~optForAlert, [~,~,gt,dt]=acfTest('name',resultFile,'imgDir',[dataDir 'test/images'],...
  'gtDir',[dataDir 'test/annotations'],'pLoad',pLoad,...
  'show',2,'color','g','reapply',0); 
end

%optionally show top false positives ('type' can be 'fp','fn','tp','dt')
if( 0 ), bbGt('cropRes',gt,dt,imgNms,'type','fn','n',50,...
    'show',3,'dims',[20.5, 50]); end

%test detector and plot roc (see acfTest)
if showBBTest && ~optForAlert
   hold on;
   [~,~,gt,dt]=acfTest('name',[resultFile 'Comb'],'imgDir',[dataDir 'test/images'],...
  'gtDir',gtDir,'pLoad',[{'lbls',{'person'},'ilbls',{'people'},'squarify',{3,.41}},...
  'hRng',[50 inf],'vRng',[.65 1],'xRng',[5 635],'yRng',[5 475]],...
  'show',2,'color','b','reapply',0);
end

%optionally show top false positives ('type' can be 'fp','fn','tp','dt')
if( 0 ), bbGt('cropRes',gt,dt,imgNms,'type','fn','n',50,...
    'show',5,'dims',[20.5, 50]); end

%% check accuracy/recall of alarms
if 0 && showAlertTest   % self labeled gt alerts
    gtFile = 'labels/gtLabels.txt';
    dtFile1 = 'results/CombineCaltechDets.txt';
    dtFile2 = 'results/CombineCaltechCombDets.txt';
    [gt,dt] = alertHelper('loadAlerts',gtFile,dtFile1);
    [prec1, tpr1, fpr1, thresh1] = alertHelper('compPlots',gt,dt,'b-x',0);
    [gt,dt] = alertHelper('loadAlerts',gtFile,dtFile2);
    [prec2, tpr2, fpr2, thresh2] = alertHelper('compPlots',gt,dt,'g-o',1);
    subplot(1,2,1);
    axis([0,1,0,1]);
    subplot(1,2,2);
    axis([0,0.1,0,1]);
end

if showAlertTest
    dtFile = [resultFile 'Dets.txt'];
    [gt,dt] = bbGt('loadAll',gtDir,dtFile,pLoad);
    % alertHelper expects label 0 == negative
    gt1=[1:length(gt); zeros(1,length(gt))]';
    for idx=1:length(gt)
        if ~isempty(gt{idx})
            gt1(idx,2)=prod(gt{idx}(:,5))==0;
        end
    end
    dt1=[1:length(dt); inf(1,length(dt))]';
    for idx=1:length(dt)
        if ~isempty(dt{idx})
            dt1(idx,2)=max(dt{idx}(:,5));
        end
    end
    dt1(dt1(:,2)==Inf,2)=min(dt1(:,2));
    
    dtFile = [resultFile 'Comb' 'Dets.txt'];
    [gt,dt] = bbGt('loadAll',gtDir,dtFile,pLoad);
    dt2=[1:length(dt); inf(1,length(dt))]';
    for idx=1:length(dt)
        if ~isempty(dt{idx})
            dt2(idx,2)=max(dt{idx}(:,5));
        end
    end
    dt2(dt2(:,2)==Inf,2)=min(dt2(:,2));
    
    [prec1, tpr1, fpr1, thresh1] = alertHelper('compPlots',gt1,dt1);
    [prec2, tpr2, fpr2, thresh2] = alertHelper('compPlots',gt1,dt2);
    
    figure;
    subplot(1,2,1);
    plot([0; tpr1], [1 ; prec1], 'b-x'); % add pseudo point to complete curve
    hold on;
    plot([0; tpr2], [1 ; prec2], 'g-o'); % add pseudo point to complete curve
    xlabel('recall');
    ylabel('precision');
    title('precision-recall graph');
    axis([0,1,0,1]);
    subplot(1,2,2);
    plot([0; fpr1], [0; tpr1], 'b-x');
    hold on;
    plot([0; fpr2], [0; tpr2], 'g-o');
    xlabel('false positive rate');
    ylabel('true positive rate');
    title('roc curve');
    axis([0,0.1,0,1]);
    rect = get(gcf,'pos');
    rect(3) = 2 * rect(3);
    set(gcf,'pos',rect);
end

end

function [bbs_res,afterHoF,afterHeF,afterThresh] = postProcess(I,model,bbs,...
    thresh_IcfIg,thresh_dpm,hMin,thresh_hMin,horizon,optForAlert)
afterThresh=0; afterHoF=0; afterHeF=0;
if size(bbs,1) == 0
    bbs_res=bbs;
    return;
end
% apply threshold to bbs
if horizon
    tmpb=size(bbs,1);
    [h,~,~]=size(I);
    lim_up = horizon + h*0.05;
    lim_down = horizon - h*0.05;
    bbs = bbs(lim_down<(bbs(:,2)+bbs(:,4)*2/3) & lim_up>(bbs(:,2)+bbs(:,4)/3),:);
end
afterHoF=size(bbs,1);
% run dpm until 1 positive match is found
bbs_res=zeros(0,5);
for bbIdx=1:size(bbs,1)
    if(thresh_hMin && bbs(bbIdx,4)<hMin)
        continue;
    end
    afterHeF = afterHeF + 1;
    if(thresh_IcfIg && bbs(bbIdx,5) > thresh_IcfIg)
        maxScore_icf = 105;
        maxScore_dpm = 2.75;
        bbs_res(end+1,:)=bbs(bbIdx,:);
        bbs_res(end,5)=((bbs_res(end,5)-thresh_IcfIg)/(maxScore_icf-thresh_IcfIg))...
                        *(maxScore_dpm-thresh_dpm) + maxScore_dpm;%thresh_dpm;
        if optForAlert, break; else continue; end
    end
    afterThresh = afterThresh + 1;
    I_bb = I(max(1,bbs(bbIdx,2)):min(size(I,1),bbs(bbIdx,2)+bbs(bbIdx,4)),...
             max(1,bbs(bbIdx,1)):min(size(I,2),bbs(bbIdx,1)+bbs(bbIdx,3)),:);
    scale = 41/size(I_bb,2);
    I_bb = imresize(I_bb,scale);
    [ds, bs] = imgdetect(I_bb, model, thresh_dpm);
    if ~isempty(bs)
        if model.type == model_types.Grammar
            bs = [ds(:,1:4) bs];
        end
        if model.type == model_types.MixStar
            % get bounding boxes
            if isfield(model, 'bboxpred')
                bbox = bboxpred_get(model.bboxpred, ds, reduceboxes(model, bs));
            else
                bbox = ds(:,[1:4,6]);
            end
            bbox = clipboxes(I, bbox);
            top = nms(bbox, 0.5);
            bbs_res(end+1,:) = bbox(top(1),:);
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
        if optForAlert, break; end
    elseif ~isempty(ds)
        top = nms(ds, 0.5);
        bbs_res(end+1,:) = ds(top,:);
        bbs_res(end,3)=bbs_res(end,3)-bbs_res(end,1);
        bbs_res(end,4)=bbs_res(end,4)-bbs_res(end,2);
        bbs_res(end,1)=bbs_res(end,1)+bbs(bbIdx,1);
        bbs_res(end,2)=bbs_res(end,2)+bbs(bbIdx,2);
        if optForAlert, break; end
    else
        % this returns the original deteciton, if dpm fails
        %bbs_res=bbs;
    end
end
end