function plotAll()
dataDir = '/media/Volume_1/capstone2/caltech_ped_dataset/data-USA/';
%dataDir = '/tmp/data-USA/';
gtDir = [dataDir 'test/annotations'];
hMin=55;
pLoad = [{'lbls',{'person'},'ilbls',{'people'},'squarify',{3,.41}},...
  'hRng',[hMin inf],'vRng',[.95 1],'xRng',[5 635],'yRng',[5 475]];

detFiles={{'results/CombineCaltechDets_0.004.txt','FPDW only','b'},
          {'results/55/CombineCaltechCombDets1_0.004_.95_-.9_23_55.txt','FPDW + DPM','g'},
          {'results/DPMCaltechDets.txt','DPM only','r'}};
 
% detFiles={{'results/CombineCaltechDets_0.004.txt','FPDW baseline','b'},
%           {'results/55/CombineCaltechCombDets1_0.004_.95_-.9_23_55.txt','FPDW+DPM - All filters','g'},
%           {'results/filtering/CombineCaltechCombDets_noFilter.txt','FPDW+DPM - No filters','c'},
%           {'results/filtering/CombineCaltechCombDets_onlyHorizon.txt','FPDW+DPM - Only horizon filter','r'},
%           {'results/filtering/CombineCaltechCombDets_onlyIgThresholdDPM.txt','FPDW+DPM - Only FPDW score filter','k'}};

% detFiles={{'results/CombineCaltechDets_0.004.txt','FPDW: threshold -0.004','b'},
%           {'results/55/CombineCaltechCombDets1_0.004_.95_-.9_23_55.txt','FPDW+DPM: FPDW threshold -0.004','g'},
%           {'results/fpdw/CombineCaltechDets_0.005.txt','FPDW: threshold -0.005','c'},
%           {'results/fpdw/CombineCaltechCombDets_0.005.txt','FPDW+DPM: FPDW threshold -0.005','r'},
%           {'results/fpdw/CombineCaltechDets_0.003.txt','FPDW: threshold -0.003','m'},
%           {'results/fpdw/CombineCaltechCombDets_0.003.txt','FPDW+DPM: FPDW threshold -0.003','k'}};
 
% detFiles={{'results/CombineCaltechDets_0.004.txt','FPDW baseline','b'},
%           {'results/55/CombineCaltechCombDets1_0.004_.95_-.9_23_55.txt','FPDW+DPM - All filters','g'},
%           {'results/dpmThresh/CombineCaltechCombDets_-0.5.txt','FPDW+DPM - No filters','c'},
%           {'results/dpmThresh/CombineCaltechCombDets_0.0.txt','FPDW+DPM - Only horizon filter','r'},
%           {'results/dpmThresh/CombineCaltechCombDets_-1.2.txt','FPDW+DPM - Only FPDW score filter','m'},
%           {'results/dpmThresh/CombineCaltechCombDets_-1.5.txt','FPDW+DPM - Optimization for alert','k'}};

figure;
subplot(1,2,1);
xlabel('recall');
ylabel('precision');
title('precision-recall graph');
axis([0,1,0,1]);
hold on;
grid on;
subplot(1,2,2);
xlabel('false positive rate');
ylabel('true positive rate');
title('roc curve');
axis([0,0.1,0,1]);
hold on;
grid on;
rect = get(gcf,'pos');
rect(3) = 2 * rect(3);
set(gcf,'pos',rect);

persistent lstringPR lstringROC
lstringPR=[]; lstringROC=[];

numPlots = length(detFiles);
for i=1:numPlots
    dtFile = [detFiles{i}{1}];
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
    
    [prec1, tpr1, fpr1, thresh1] = alertHelper('compPlots',gt1,dt1);
    subplot(1,2,1);
    plot([0; tpr1], [1 ; prec1], detFiles{i}{3});
    lstringPR{end+1} = detFiles{i}{2};
    legend(lstringPR, 'Location', 'Best');
    hold on;
    subplot(1,2,2);
    plot([0; fpr1], [0; tpr1], detFiles{i}{3});
    lstringROC{end+1} = detFiles{i}{2};
    legend(lstringROC, 'Location', 'Best');
    hold on;
end