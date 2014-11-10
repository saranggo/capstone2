function [pos, neg, impos] = caltech_data(conf)
% Get training data from the caltech dataset.
%   [pos, neg, impos] = caltech_data(cls, year)
%
% Return values
%   pos     Each positive example on its own
%   neg     Each negative image on its own
%   impos   Each positive image with a list of foreground boxes

cachedir   = conf.paths.model_dir;

try
  load([cachedir 'cache_caltech_1']);
catch
  Is1 = sampleWins(conf, 1, 0);
  Is0 = sampleWins(conf, 0, 0);
  % Positive examples from the foreground dataset
  pos      = [];
  impos    = [];
  numpos   = 0;
  numimpos = 0;
  dataid   = 0;
  for i = 1:length(Is1);
    tic_toc_print('caltech: parsing positives\n');
    % Parse record and exclude difficult examples
    Inm           = Is1{i}{1};
    exactinds     = Is1{i}{2}(Is1{i}{2}(:,5)==0,:);
    count         = size(exactinds,1);
    % Skip if there are no objects in this image
    if count == 0
      continue;
    end

    % Create one entry per bounding box in the pos array
    for j = 1:count
      numpos = numpos + 1;
      dataid = dataid + 1;
      bbox   = exactinds(j,1:4);
      
      pos(numpos).im      = Inm;
      pos(numpos).x1      = bbox(1);
      pos(numpos).y1      = bbox(2);
      pos(numpos).x2      = bbox(3);
      pos(numpos).y2      = bbox(4);
      pos(numpos).boxes   = bbox;
      pos(numpos).flip    = false;
      pos(numpos).trunc   = [];
      pos(numpos).dataids = dataid;
      pos(numpos).sizes   = (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1);

      % Create flipped example
      numpos  = numpos + 1;
      dataid  = dataid + 1;
      oldx1   = bbox(1);
      oldx2   = bbox(3);
      bbox(1) = Is1{i}{3}(2) - oldx2 + 1;
      bbox(3) = Is1{i}{3}(2) - oldx1 + 1;

      pos(numpos).im      = Inm;
      pos(numpos).x1      = bbox(1);
      pos(numpos).y1      = bbox(2);
      pos(numpos).x2      = bbox(3);
      pos(numpos).y2      = bbox(4);
      pos(numpos).boxes   = bbox;
      pos(numpos).flip    = true;
      pos(numpos).trunc   = [];
      pos(numpos).dataids = dataid;
      pos(numpos).sizes   = (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1);
    end

    % Create one entry per foreground image in the impos array
    numimpos                = numimpos + 1;
    impos(numimpos).im      = Inm;
    impos(numimpos).boxes   = zeros(count, 4);
    impos(numimpos).dataids = zeros(count, 1);
    impos(numimpos).sizes   = zeros(count, 1);
    impos(numimpos).flip    = false;

    for j = 1:count
      dataid = dataid + 1;
      bbox   = exactinds(j,1:4);
      
      impos(numimpos).boxes(j,:) = bbox;
      impos(numimpos).dataids(j) = dataid;
      impos(numimpos).sizes(j)   = (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1);
    end

    % Create flipped example
    numimpos                = numimpos + 1;
    impos(numimpos).im      = Inm;
    impos(numimpos).boxes   = zeros(count, 4);
    impos(numimpos).dataids = zeros(count, 1);
    impos(numimpos).sizes   = zeros(count, 1);
    impos(numimpos).flip    = true;
    unflipped_boxes         = impos(numimpos-1).boxes;
    
    for j = 1:count
      dataid  = dataid + 1;
      bbox    = unflipped_boxes(j,:);
      oldx1   = bbox(1);
      oldx2   = bbox(3);
      bbox(1) = Is1{i}{3}(2) - oldx2 + 1;
      bbox(3) = Is1{i}{3}(2) - oldx1 + 1;

      impos(numimpos).boxes(j,:) = bbox;
      impos(numimpos).dataids(j) = dataid;
      impos(numimpos).sizes(j)   = (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1);
    end
  end

  % Negative examples from the background dataset
  neg    = [];
  numneg = 0;
  for i = 1:length(Is0);
    tic_toc_print('caltech: parsing negatives\n');
    Inm                = Is0{i}{1};
    dataid             = dataid + 1;
    numneg             = numneg+1;
    neg(numneg).im     = Inm;
    neg(numneg).flip   = false;
    neg(numneg).dataid = dataid;
  end
  
  save([cachedir 'cache_caltech_1'], 'pos', 'neg', 'impos');
end
end

function Is = sampleWins(conf, positive, stage)
% Load or sample windows for training detector.
start=clock;
if( positive ), n=conf.training.num_positives; else n=conf.training.num_negatives_large; end
if( positive ), crDir=conf.paths.posWinDir; else crDir=conf.paths.negWinDir; end
if( exist(crDir,'dir') && stage==0 )
  % if window directory is specified simply load windows
  fs=bbGt('getFiles',{crDir}); nImg=length(fs); assert(nImg>0);
  if(nImg>n), fs=fs(:,randSample(nImg,n)); end; n=nImg;
  for i=1:n, fs{i}=[{conf.imreadf},fs(i),conf.imreadp]; end
  Is=cell(1,n); parfor i=1:n, Is{i}=feval(fs{i}{:}); end
else
  % sample windows from full images using sampleWins1()
  hasGt=positive||isempty(conf.paths.negImgDir);
  if(hasGt), fs={conf.paths.posImgDir,conf.paths.posGtDir};
  else fs={conf.paths.negImgDir,conf.paths.posGtDir}; end
  fs=bbGt('getFiles',fs); nImg=size(fs,2); assert(nImg>0);
  if(~isinf(n)), fs=fs(:,randperm(nImg)); end; Is=cell(nImg*1000,1);
  tid=ticStatus('Sampling windows',1,30); k=0; kbb=0; i=0; batch=64;
  while( i<nImg && kbb<n && k<n )
    batch=min(batch,nImg-i); Is1=cell(1,batch);
    for j=1:batch, ij=i+j;
      %I = feval(conf.imreadf,fs{1,ij},conf.imreadp{:}); %#ok<P_F_B_N_S>
      %gt=[]; if(hasGt), 
      [~,gt]=bbGt('bbLoad',fs{2,ij},conf.pLoad);
      gt=gt(gt(:,5)==0,:);
      if(positive)
          r=conf.modelDs(2)/conf.modelDs(1); 
          try assert(all(abs(gt(:,3)./gt(:,4)-r)<1e-5)); 
          catch, disp('ratio error'), end
          if ~isempty(gt), Is1{j}={fs{1,ij},bbs_convert(gt),[480 640 3]};
          else Is1{j}={}; end
          %Is1{j} = sampleWins1( fs{1,ij}, conf, gt, stage, positive );
      else
          if isempty(gt), Is1{j}={fs{1,ij},bbs_convert(gt)}; else Is1{j}={}; end
      end
    end
    Is1=Is1(~cellfun(@isempty,Is1)); k1=length(Is1); Is(k+1:k+k1)=Is1; k=k+k1;
    for Is11=Is1, kbb=kbb+size(Is11{:}{2},1); end
    %if(k>n), Is=Is(randSample(k,n)); k=n; end
    i=i+batch; tocStatus(tid,min(1,max(i/nImg,k/n)));
  end
  Is=Is(1:k); fprintf('Sampled %i windows from %i images.\n',kbb,i);
end
% optionally jitter positive windows
if(length(Is)<2), Is={}; return; end
%nd=ndims(Is{1})+1; Is=cat(nd,Is{:});
if( positive && isstruct(conf.pJitter) )
  conf.pJitter.hasChn=(nd==4); Is=jitterImage(Is,conf.pJitter);
  ds=size(Is); ds(nd)=ds(nd)*ds(nd+1); Is=reshape(Is,ds(1:nd));
end
fprintf('Done sampling windows (time=%.0fs).\n',etime(clock,start));
end

% function Is = sampleWins1( Inm, conf, gt, stage, positive )
% % Sample windows from I given its ground truth gt.
% modelDs=conf.modelDs; modelDsPad=conf.modelDsPad;
% %I = feval(conf.imreadf,Inm,conf.imreadp{:});  
% if( positive ), bbs=gt; bbs=bbs(bbs(:,5)==0,:); 
% else 
%   if( stage==0 )
%     % generate candidate bounding boxes in a grid
%     %[h,w,~]=size_I; 
%     h=480; w=640; h1=modelDs(1); w1=modelDs(2);
%     n=conf.nPerNeg; ny=sqrt(n*h/w); nx=n/ny; ny=ceil(ny); nx=ceil(nx);
%     [xs,ys]=meshgrid(randi(w-floor(modelDs(2)),1,nx),randi(h-floor(modelDs(1)),1,ny));
%     bbs=[xs(:) ys(:)]; bbs(:,3)=w1; bbs(:,4)=h1; bbs=bbs(1:n,:);
%   else
%     % run detector to generate candidate bounding boxes
%     I = feval(conf.imreadf,Inm,conf.imreadp{:});
%     bbs=acfDetect(I,conf.acfDetector); [~,ord]=sort(bbs(:,5),'descend');
%     bbs=bbs(ord(1:min(end,conf.nPerNeg)),1:4);
%   end
%   if( ~isempty(gt) )
%     % discard any candidate negative bb that matches the gt
%     n=size(bbs,1); keep=false(1,n);
%     for i=1:n, keep(i)=all(bbGt('compOas',bbs(i,:),gt,gt(:,5))<.1); end
%     bbs=bbs(keep,:);
%   end
% end
% % grow bbs to a large padded size and finally crop windows
% %modelDsBig=max(8*shrink,modelDsPad)+max(2,ceil(64/shrink))*shrink;
% r=modelDs(2)/modelDs(1); 
% try assert(all(abs(bbs(:,3)./bbs(:,4)-r)<1e-5)); 
% catch, disp('ratio error'), end
% % TODO : is padding required?
% %r=modelDsPad./modelDs; bbs=bbApply('resize',bbs,r(1),r(2));
% %Is=bbApply('crop',I,bbs,'replicate',modelDsPad([2 1]));
% if ~isempty(bbs), Is={Inm,bbs_convert(bbs),[480 640 3]};
% else Is={}; end
% end

function bbs = bbs_convert(bbs)
% tmp=bbs(:,1);
% bbs(:,1)=bbs(:,2);
% bbs(:,2)=tmp;
% tmp=bbs(:,3);
% bbs(:,3)=bbs(:,4);
% bbs(:,4)=tmp;
bbs(:,3)=bbs(:,1)+bbs(:,3);
bbs(:,4)=bbs(:,2)+bbs(:,4);
end