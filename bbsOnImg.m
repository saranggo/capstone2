imgDir = '/home/gnaras/ws_cap/caltech_ped_dataset/data-USAtrain/images';
gtDir = '/home/gnaras/ws_cap/caltech_ped_dataset/data-USAtrain/annotations';
pLoad=[{'lbls',{'person'},'ilbls',{'people'},'squarify',{3,.41}},...
    'hRng',[50 inf],'vRng',[.65 1],'xRng',[5 635],'yRng',[5 475]];
imgNms = bbGt('getFiles',{imgDir});
[gt,~] = bbGt('loadAll',gtDir,[],pLoad);
for imgIdx=1:size(imgNms,2)
    I=imread(imgNms{imgIdx}); figure(1); im(I);
%     if ~isempty(gt{imgIdx}) && gt{imgIdx}(5)==0
    bbApply('draw',[gt{imgIdx}],'g'); 
%     end
    pause(.1);
end