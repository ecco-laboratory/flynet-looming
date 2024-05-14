%% load libraries
% it seems a little idiotic to do this inside the function definition 
% but matlab appears not to allow function scripts to have ANY other code before the function
addpath('/home/data/eccolab/Code/GitHub/spm12/spm12'); % per spm docs, do not genpath it
addpath(genpath('/home/data/eccolab/Code/GitHub/CanlabCore/CanlabCore'));
addpath(genpath('/home/data/eccolab/Code/GitHub/Neuroimaging_Pattern_Masks'));

%% variables that must be specified before the script is called
% project_dir
% data_subdir
% sub_prefix
% roi
% out_name

%% actually do the processing
% this is made to be called from targets, so it just naked does the stuff
% as opposed to defining a matlab function to be called from within matlab

DATA = [];
% set the ROI at the beginning of the pass because it doesn't need to change inside the loops
% ROI options in the canlab tools I think: 'Ctx_V1', 'Bstem_SC'
% If you want to load subject-specific masks, you need to use fmri_data()
this_atlas_roi = select_atlas_subset(load_atlas('canlab2018'),{roi});

all_possible_subs = dir(fullfile(project_dir, strcat(sub_prefix, '*')));
all_valid_subs = {};
for sub_idx=1:length(all_possible_subs)
    % if the retinotopy folder exists for this subject
    % put them on the smaller list of valid subjects
    if exist(fullfile(project_dir, all_possible_subs(sub_idx).name, data_subdir), 'dir') == 7
        all_valid_subs{end+1} = all_possible_subs(sub_idx).name;
    end
end
% if the subjects array counts up continuously, subject num could be used instead of the counter
% to index the array slice corresponding to that subject's data
% if it does not,
% for loop over the counter variable and set subject within each loop?
%%
% must loop over this subject index variable because the actual subject IDs
% of those who have retinotopy data are not contiguous
n_valid_data_subs = 1; 
for sub_idx=1:length(all_valid_subs)
    
    % deleteme mt: cd(fullfile(project_dir, all_valid_subs{sub_idx}, data_subdir))

    % I think Phil made the w versions of the files? They're preprocessed somehow?
    % Be mindful that these will be in FILENAME ALPHABETICAL ORDER!!!
    files = dir(fullfile(project_dir, all_valid_subs{sub_idx}, data_subdir, 'w*.nii'));
    data_all_runs = [];
    
    for f=1:length(files)
        data=fmri_data(fullfile(files(f).folder, files(f).name));
        %todo - simple preproc with detrending, filter, and motion regression
        data.X = readmatrix(fullfile(files(f).folder, ['rp_' files(f).name(2:end-4) '.txt']));
        % currently only band pass filtering. Too tired for more
        %data = canlab_connectivity_preproc(data, 'hpf', .008, 2);
        data = canlab_connectivity_preproc(data,'bpf',[.667/32 2/32],2);
        masked_dat = apply_mask(data, this_atlas_roi);
        % the second dim of DATA is the number of voxels
        data_all_runs(:, :, f) = masked_dat.dat;
    end
    
    % will fail SILENTLY for the one subject with 12 bonky V1 voxels
    try
        DATA(n_valid_data_subs, :, :, :) = data_all_runs;
    catch
        continue
    end
    % must track this separately from sub_idx because of that one subject with 12 bonky V1 voxels
    n_valid_data_subs = n_valid_data_subs + 1;
end

% it appears I cannot non-standard-eval to name the resulting data variable the same name as the output file
% and we must save it as .mat, which doesn't let us force-load the underlying variables into preset names, to preserve the 4 dimensions
% so we will just have to standardly assume that any .mat file saved by this script contains one matrix called DATA
save(out_name, 'DATA')
