function gtLabeler()
init_env;
dataDir = '/media/Volume_1/capstone2/caltech_ped_dataset/data-USA/';
resultFile = 'labels/gtLabels.txt';
pLoad={'lbls',{'person'},'ilbls',{'people'},'squarify',{3,.41}};

d=fileparts(resultFile); if(~isempty(d)&&~exist(d,'dir')), mkdir(d); end
[gtbbs,~] = bbGt('loadAll',[dataDir 'test/annotations'],[],pLoad);
imgNms=bbGt('getFiles',{[dataDir 'test/images']});
n = length(imgNms);
gt = zeros(n,1);
for imgIdx=12512:n
    I=imread(imgNms{imgIdx}); 
    figure(1); im(I);
    title(imgIdx);
    bbApply('draw',gtbbs{imgIdx});
    if ~isempty(gtbbs{imgIdx})
        reply = input('Alert? 1/0 [0] ', 's');
        if isempty(reply)
            reply = '0';
        end
        if reply == '1'
            gt(imgIdx) = 1; 
        end
    end
    dlmwrite(resultFile,[imgIdx,gt(imgIdx)],'-append');
end
