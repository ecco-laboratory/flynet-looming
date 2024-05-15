%% load libraries
% it seems a little idiotic to do this inside the function definition 
% but matlab appears not to allow function scripts to have ANY other code before the function
addpath('../ignore/libraries/spm12'); % per spm docs, do not genpath it
addpath(genpath('../ignore/libraries/CanlabCore/CanlabCore'));
addpath(genpath('../ignore/libraries/Neuroimaging_Pattern_Masks'));

%% variables that must be specified before the script is called
% roi: in NeuroimagingPatternMasks syntax, in this case one of: 'Ctx_V1', 'Bstem_SC'
% all_files: a cell array of cell arrays containing full paths to SPM-preprocessed niftis
% the outer dimension corresponds to subject
% the inner dimension corresponds to run file
% out_name: the filename for the _single_ output file containing data from all subs/runs

%% actually do the processing
% this is made to be called from targets, so it just naked does the stuff
% as opposed to defining a matlab function to be called from within matlab

DATA = [];
this_atlas_roi = select_atlas_subset(load_atlas('canlab2018'),{roi});

for sub_idx=1:length(all_files)

    % Be mindful that these are expected in FILENAME ALPHABETICAL ORDER!!!
    files = all_files{sub_idx};
    data_all_runs = [];
    
    for f=1:length(files)
        [path, filename, ~] = fileparts(files{f});
        data=fmri_data(files{f});
        % simple preproc with detrending, filter, and motion regression
        % these files are output by SPM realign/normalize along with the realigned niftis
        % you must trust that they will be there if the realign batch has run
        data.X = readmatrix(fullfile(path, ['rp_' filename(2:end) '.txt']));
        % currently only band pass filtering. Too tired for more
        data = canlab_connectivity_preproc(data,'bpf',[.667/32 2/32],2);
        masked_dat = apply_mask(data, this_atlas_roi);
        % the second dim of DATA is the number of voxels
        data_all_runs(:, :, f) = masked_dat.dat;
    end
    
    % will fail SILENTLY for the one subject with 12 bonky V1 voxels
    try
        DATA(sub_idx, :, :, :) = data_all_runs;
    catch
        continue
    end

end

% it appears I cannot non-standard-eval to name the resulting data variable the same name as the output file
% and we must save it as .mat, which doesn't let us force-load the underlying variables into preset names, to preserve the 4 dimensions
% so we will just have to standardly assume that any .mat file saved by this script contains one matrix called DATA
save(out_name, 'DATA')
