function plotAll()
dataDir = '/media/Volume_1/capstone2/caltech_ped_dataset/data-USA/';
gtDir = [dataDir 'test/annotations'];
hMin=55;
pLoad = [{'lbls',{'person'},'ilbls',{'people'},'squarify',{3,.41}},...
  'hRng',[hMin inf],'vRng',[.65 1],'xRng',[5 635],'yRng',[5 475]];
detFiles={{'results/CombineCaltechDets_0.004.txt','FPDW only','b'},
          {'results/CombineCaltechCombDets1_23_55_opt_1scale.txt','FPDW + DPM','g'}}
      
figure;
subplot(1,2,1);
xlabel('recall');
ylabel('precision');
title('precision-recall graph');
axis([0,1,0,1]);
grid on;
hold on;
subplot(1,2,2);
xlabel('false positive rate');
ylabel('true positive rate');
title('roc curve');
axis([0,0.1,0,1]);
grid on;
hold on;
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
    legend(lstringPR);
    hold on;
    subplot(1,2,2);
    plot([0; fpr1], [0; tpr1], detFiles{i}{3});
    lstringROC{end+1} = detFiles{i}{2};
    legend(lstringROC);
    hold on;
end