function model = dpmTrainCaltech()
init_env;
dataDir = '/media/Volume_1/capstone2/caltech_ped_dataset/data-USA/';
timestamp = datestr(datevec(now()), 'dd.mmm.yyyy:HH.MM.SS');

% load acf detector
t=load('models/AcfCaltechDetector');
detector = t.detector;
detector = acfModify(detector,'cascThr',-1,'cascCal',-.005);

conf = [];
conf = cv(conf, 'acfDetector', detector);
conf = cv(conf, 'single_byte_size', 4);
conf = cv(conf, 'imreadf', @imread);
conf = cv(conf, 'imreadp', {});

conf = cv(conf, 'pLoad', [{'lbls',{'person'},'ilbls',{'people'},'squarify',{3,.41}} 'hRng',[50 inf], 'vRng',[1 1] ]);
%conf = cv(conf, 'pJitter', struct('flip',1));
conf = cv(conf, 'pJitter', '');
conf = cv(conf, 'nPerNeg', 1);
conf = cv(conf, 'modelDs', [50, 20.5]);
conf = cv(conf, 'modelDsPad', [64, 32]);

% -------------------------------------------------------------------
% Path configuration 
% -------------------------------------------------------------------
% Directory for caching models, intermediate data, and results
% [was called 'cachedir' in previous releases]
conf = cv(conf, 'paths.model_dir', 'models/');
exists_or_mkdir(conf.paths.model_dir);
conf = cv(conf, 'paths.posWinDir', []);
conf = cv(conf, 'paths.negWinDir', []);
conf = cv(conf, 'paths.negImgDir', [dataDir 'train/images']);
conf = cv(conf, 'paths.posImgDir', [dataDir 'train/images']);
conf = cv(conf, 'paths.posGtDir', [dataDir 'train/annotations']);

% -------------------------------------------------------------------
% Training configuration 
% -------------------------------------------------------------------
conf = cv(conf, 'training.C', 0.001);
conf = cv(conf, 'training.bias_feature', 10);
% File size limit for the feature vector cache (2^30 bytes = 1GB)
conf = cv(conf, 'training.cache_byte_limit', 3*2^30);
% Location of training log (matlab diary)
conf.training.log = @(x) sprintf([conf.paths.model_dir '%s.log'], x);

conf = cv(conf, 'training.cache_example_limit', 24000);
conf = cv(conf, 'training.num_negatives_small', 1000);
conf = cv(conf, 'training.num_negatives_large', 10000);
conf = cv(conf, 'training.num_positives', Inf);
conf = cv(conf, 'training.wlssvm_M', 0);
conf = cv(conf, 'training.fg_overlap', 0.7);

conf = cv(conf, 'training.lbfgs.options.verbose', 2);
conf = cv(conf, 'training.lbfgs.options.maxIter', 1000);
conf = cv(conf, 'training.lbfgs.options.optTol', 0.000001);

conf = cv(conf, 'training.interval_fg', 5);
conf = cv(conf, 'training.interval_bg', 4);

% -------------------------------------------------------------------
% Evaluation configuration 
% -------------------------------------------------------------------
conf = cv(conf, 'eval.interval', 10);
conf = cv(conf, 'eval.test_set', 'test');
conf = cv(conf, 'eval.max_thresh', -1.1);

% -------------------------------------------------------------------
% Feature configuration 
% -------------------------------------------------------------------
conf = cv(conf, 'features.sbin', 8);
conf = cv(conf, 'features.dim', 32);
conf = cv(conf, 'features.truncation_dim', 32);
conf = cv(conf, 'features.extra_octave', false);

% -------------------------------------------------------------------
% Cascade configuration 
% -------------------------------------------------------------------
conf = cv(conf, 'cascade.data_dir', [pwd() '/star-cascade/data/']);
exists_or_mkdir(conf.cascade.data_dir);

% Record a log of the training and test procedure
diary(conf.training.log(['caltech-' timestamp]));


seed_rand();
cachedir = conf.paths.model_dir;

% Load the training data
[pos, neg, impos] = caltech_data(conf);

n=1;
cls='caltechped';
note='caltech pedestrian dataset';
% Split foreground examples into n groups by aspect ratio
spos = split(pos, n);

max_num_examples = conf.training.cache_example_limit;
num_fp           = conf.training.wlssvm_M;
fg_overlap       = conf.training.fg_overlap;

% Select a small, random subset of negative images
% All data mining iterations use this subset, except in a final
% round of data mining where the model is exposed to all negative
% images
num_neg   = length(neg);
neg_perm  = neg(randperm(num_neg));
neg_small = neg_perm(1:min(num_neg, conf.training.num_negatives_small));
neg_large = neg; % use all of the negative images

% Train one asymmetric root filter for each aspect ratio group
% using warped positives and random negatives
try
  load([cachedir cls '_lrsplit1']);
catch
  seed_rand();
  for i = 1:n
    models{i} = root_model_c(conf, cls, spos{i}, note);
    % Split the i-th aspect ratio group into two clusters: 
    % left vs. right facing instances
    inds = lrsplit_c(conf, models{i}, spos{i});
    % Train asymmetric root filter on one of these groups
    models{i} = train_c(conf, models{i}, spos{i}(inds), neg_large, true, true, 1, 1, ...
                      max_num_examples, fg_overlap, 0, false, ...
                      ['lrsplit1_' num2str(i)]);
  end
  save([cachedir cls '_lrsplit1'], 'models');
end

% Train a mixture of two root filters for each aspect ratio group
% Each pair of root filters are mirror images of each other
% and correspond to two latent orientations choices
% Training uses latent positives and hard negatives
try
  load([cachedir cls '_lrsplit2']);
catch
  seed_rand();
  for i = 1:n
    % Build a mixture of two (mirrored) root filters
    models{i} = lr_root_model(models{i});
    models{i} = train_c(conf, models{i}, spos{i}, neg_small, false, false, 4, 3, ...
                      max_num_examples, fg_overlap, 0, false, ...
                      ['lrsplit2_' num2str(i)]);
  end
  save([cachedir cls '_lrsplit2'], 'models');
end

% Train a mixture model composed all of aspect ratio groups and 
% latent orientation choices using latent positives and hard negatives
try 
  load([cachedir cls '_mix']);
catch
  seed_rand();
  % Combine separate mixture models into one mixture model
  model = model_merge(models);
  model = train_c(conf, model, impos, neg_small, false, false, 1, 5, ...
                max_num_examples, fg_overlap, num_fp, false, 'mix');
  save([cachedir cls '_mix'], 'model');
end

% Train a mixture model with 2x resolution parts using latent positives
% and hard negatives
try 
  load([cachedir cls '_parts']);
catch
  seed_rand();
  % Add parts to each mixture component
  for i = 1:2:2*n
    % Top-level rule for this component
    ruleind = i;
    % Top-level rule for this component's mirror image
    partner = i+1;
    % Filter to interoplate parts from
    filterind = i;
    model = model_add_parts(model, model.start, ruleind, ...
                            partner, filterind, 8, [6 6], 1);
    % Enable learning location/scale prior
    bl = model.rules{model.start}(i).loc.blocklabel;
    model.blocks(bl).w(:)     = 0;
    model.blocks(bl).learn    = 1;
    model.blocks(bl).reg_mult = 1;
  end
  % Train using several rounds of positive latent relabeling
  % and data mining on the small set of negative images
  model = train_c(conf, model, impos, neg_small, false, false, 8, 10, ...
                max_num_examples, fg_overlap, num_fp, false, 'parts_1');
  % Finish training by data mining on all of the negative images
  model = train_c(conf, model, impos, neg_large, false, false, 1, 5, ...
                max_num_examples, fg_overlap, num_fp, true, 'parts_2');
  save([cachedir cls '_parts'], 'model');
end

save([cachedir cls '_final'], 'model');

method='default';
model = bboxpred_train_c(conf, cls, method);
save([cachedir cls '_final_bbp'], 'model');
end



% -------------------------------------------------------------------
% Helper functions
% -------------------------------------------------------------------

% -------------------------------------------------------------------
% Make directory path if it does not already exist.
function made = exists_or_mkdir(path)
    made = false;
    if exist(path) == 0
      unix(['mkdir -p ' path]);
      made = true;
    end
end

% -------------------------------------------------------------------
% Does nothing if conf.key exists, otherwise sets conf.key to val
function conf = cv(conf, key, val)
    try
      eval(['conf.' key ';']);
    catch
      eval(['conf.' key ' = val;']);
    end
end
