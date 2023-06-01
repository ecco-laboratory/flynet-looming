%% load libraries
addpath(genpath('/home/data/eccolab/Code/GitHub/spm12'));
addpath(genpath('/home/data/eccolab/Code/GitHub/CanlabCore/CanlabCore'));
addpath(genpath('/home/data/eccolab/Code/GitHub/Neuroimaging_Pattern_Masks'));

%% specify project paths
studyforrest_dir = '/home/data/eccolab/studyforrest-data-phase2';
studyforrest_subdir = 'ses-localizer/func';
nsd_dir = '/home/data/shared/NSD/nsddata_timeseries/ppdata';
nsd_subdir = 'func1mm/timeseries';

%% actually load the fMRI data
% data_studyforrest_sc = load_fmri_data(studyforrest_dir, studyforrest_subdir, 'sub-', 'Bstem_SC');
% disp('studyforrest SC data loaded')
data_studyforrest_v1 = load_fmri_data(studyforrest_dir, studyforrest_subdir, 'sub-', 'Ctx_V1');
disp('studyforrest V1 data loaded')
%data_nsd_sc = load_fmri_data(nsd_dir, nsd_subdir, 'subj', 'Bstem_SC');
%disp('NSD SC data loaded')
%data_nsd_v1 = load_fmri_data(nsd_dir, nsd_subdir, 'subj', 'Ctx_V1');
%disp('NSD V1 data loaded')

%% save
% save('/home/mthieu/bold_retinotopy_studyforrest_sc.mat', 'data_studyforrest_sc')
save('/home/mthieu/bold_retinotopy_studyforrest_v1.mat', 'data_studyforrest_v1')
% save('/home/mthieu/bold_retinotopy_nsd_sc.mat', 'data_nsd_sc')
% save('/home/mthieu/bold_retinotopy_nsd_v1.mat', 'data_nsd_v1')

exit

%% funky function definition
function DATA = load_fmri_data(project_dir, data_subdir, sub_prefix, roi)
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
    for sub_idx=1:length(all_valid_subs)
        
        cd(fullfile(project_dir, all_valid_subs{sub_idx}, data_subdir))

        % I think Phil made the w versions of the files? They're preprocessed somehow?
        % Be mindful that these will be in FILENAME ALPHABETICAL ORDER!!!
        files = dir('w*.nii');

        for f=1:length(files)
            data=fmri_data(files(f).name);
            %todo - simple preproc with detrending, filter, and motion regression
            data.X = dlmread(['rp_' files(f).name(2:end-4) '.txt']);
            % currently only band pass filtering. Too tired for more
            %data = canlab_connectivity_preproc(data, 'hpf', .008, 2);
            data = canlab_connectivity_preproc(data,'bpf',[.667/32 2/32],2);
            masked_dat = apply_mask(data, this_atlas_roi);

            DATA(sub_idx,:,:,f) = masked_dat.dat;
        end
    end

end
