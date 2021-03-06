% Prepares your matlab workspace for using voc-release5.
global V_STARTUP;

if isempty(V_STARTUP)
	V_STARTUP = true;

	% Avoiding addpath(genpath('.')) because .git includes
	% a VERY large number of subdirectories, which makes 
	% startup slow
	incl = {'context', 'bbox_pred', 'fv_cache', ...
		  'bin', 'gdetect', 'utils', ...
		  'car_grammar', 'person_grammar', ...
		  'model', 'features', 'vis', ...
		  'data', 'train', 'test', ...
		  'external', 'star-cascade'};
	for i = 1:length(incl)
		addpath(genpath(incl{i}));
	end
	conf = voc_config();
	fprintf('%s is set up\n', conf.version);
	clear conf i incl;
	
	if ispc
		%piotr toolbox
		addpath(genpath('../toolbox'));
	else
		%piotr toolbox
		addpath(genpath('../toolbox'));

		fprintf('compiling the code...');
		compile;
		fprintf('done.\n\n');
	end
end
