% Prepares your matlab workspace for using voc-release5.
global C_STARTUP;

if isempty(C_STARTUP)
	addpath(genpath('toolbox'));
    addpath(genpath('caltech_ped_dataset_code'));
    addpath(genpath('voc-release5'));
    
	conf = voc_config();
	fprintf('%s is set up\n', conf.version);
	clear conf i incl;
	
	if ~ispc
		cd voc-release5;
        addpath(genpath('fv_cache'));
        addpath(genpath('star-cascade'));
        fprintf('compiling the code...');
		compile;
		fprintf('done.\n\n');
        cd ..;
    end
    C_STARTUP = true;
end

