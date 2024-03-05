%% setup
% library things
addpath('/home/data/eccolab/Code/GitHub/spm12/spm12');
addpath(genpath('/home/data/eccolab/Code/GitHub/CanlabCore/CanlabCore'));

% paths that it's cleaner to set up top?
studyforrest_dir = '/home/data/eccolab/studyforrest-data-phase2';

%% variables that must be set in targets before the script is called
% sc_data_path: the output file of preproc_mask_fmri_data_canlabtools; the variable in here is called DATA
% studyforrest_activation_path: convolved activations, CSV FROM R
% out_fstring: a printf-compatible format string that will be used for output filenames

%% EMERGENCY SETTING THOSE VARIABLES IF NOT RUNNING FROM TARGETS
% sc_data_path = fullfile(studyforrest_dir, 'fmri_data_canlabtooled_sc.mat');
% studyforrest_activation_path = fullfile(studyforrest_dir, 'flynet_convolved_timecourses.csv');
% out_fstring = '%smap_flynet_connectivity_contrast.nii';

%% load in data from targets paths
load(sc_data_path)

% order in DATA variable: ccw, clw, con, exp
% targets now outputs this csv in this same order that the canlabtools matlab scripts read/write the data
% aka studyforrest file condition name order
% must use stable flag for unique to return the categories in data order
t = readtable(studyforrest_activation_path);
tasknames = unique(t{:,1}, 'stable');
% in units of TRs
run_length = height(t)/length(tasknames);
% we need these later for block permutation
n_cycles = 5;
cycle_length = run_length/n_cycles;

%% read in whole-brain fmri data
% get valid subjects
subs = dir(fullfile(studyforrest_dir, 'sub-*'));
n_valid_subs=0;
for s=1:length(subs)
    % in a try-catch statement so it will only read in data from subjects who did this task
    % but not specifying a priori which subjects those are
    try
        % deleteme mt: cd(fullfile(studyforrest_dir, subs(s).name, 'ses-localizer', 'func'))
        % if this subject doesn't have this scan, files will have length 0
        files = dir(fullfile(studyforrest_dir, subs(s).name, 'ses-localizer', 'func', 'w*.nii'));
        if isempty(files)
            continue
        end
        % tracks valid subjects
        n_valid_subs=n_valid_subs+1;

        for f=1:length(files) % for each of the stimulus runs
            data=fmri_data(fullfile(files(f).folder, files(f).name));
            %todo - simple preproc with detrending, filter, and motion regression
            % TODO: targets-track these txt files. are they motion regressors?
            % and find/track the script that outputs them
            data.X = readmatrix(fullfile(files(f).folder, ['rp_' files(f).name(2:end-4) '.txt']));
            data = canlab_connectivity_preproc(data,'bpf',[.667/32 2/32],2);
            % no masking. whole brain
            
            WB_DATA(n_valid_subs,:,:,f) = data.dat;
        end

    catch
    end
end

%% run phil's favorite plsregress... ONLY on each run/stim type separately
for tr=1:length(tasknames)

    % step 1: pull out this task's FlyNet activations
    % start from 3 bc the first column is condition and the second column is TR number
    for i=3:width(t)
         X(:,i-2)=t{strcmp(t{:,1},tasknames{tr}),i};
    end

    % step 2: pull out each subject's brain timecourses for this task
    % for both SC (cDAT) and whole-brain (cDAT_WB)
    % SC is for fitting the PLS
    % whole brain is for the model-based connectivity
    cDAT = []; cDAT_WB = []; sub=[];
    for s=1:n_valid_subs
        fprintf('Preparing data: run %s, fold %d\n', tasknames{tr}, s)
        % the timecourses come in with some un-stimulated TRs so we can't just go 1:run_length
        cDAT = [cDAT; zscore(squeeze(DATA(s,:,(1:run_length)+2,tr))')];
        cDAT_WB = [cDAT_WB; zscore(squeeze(WB_DATA(s,:,(1:run_length)+2,tr))')];
        sub = [sub;s*ones(run_length,1)];
    end

    % step 3:
    % fit each PLS cross-validation fold
    % leave-one-subject-out
    nc = 20;
    for s = 1:n_valid_subs
        fprintf('Fitting PLS: run %s, fold %d\n', tasknames{tr}, s)
        [~,~,~,~,b]=plsregress(repmat(X(1:run_length,:),n_valid_subs-1,1),cDAT(sub~=s,:),nc);
        % don't touch this! yhat is an output of plsregress
        yhat(sub==s,:)=[ones(length(find(sub==s)),1) X(1:run_length,:)]*b;
        % average the SC-pred response across "SC voxels" and then correlate with each whole-brain voxel
        % we now need to save the mean y-hats out for the permutation testing later as well
        yhat_all(s,tr,:)=mean(yhat(sub==s,:),2);
        vox_corr_wb(s,tr,:)=corr(squeeze(yhat_all(s,tr,:)),cDAT_WB(sub==s,:));
    end
    % originally this was getting performance at different numbers of components but we are just doing 20 now
    % perf(tr)=mean(cv_est(:,tr));
end

% get the contrast between expand and the mean of the other 3
vox_corr_wb_notexpand = squeeze(mean(vox_corr_wb(:,1:3,:), 2, "omitnan"));
vox_corr_wb_expand = squeeze(vox_corr_wb(:,4,:));
vox_corr_wb_diff = vox_corr_wb_expand - vox_corr_wb_notexpand;

% average across xval folds for the statmaps later
mean_vox_corr_wb_expand = squeeze(mean(vox_corr_wb_expand, "omitnan"));
mean_vox_corr_wb_diff = squeeze(mean(vox_corr_wb_diff, "omitnan"));
% if they were ALL NaN (voxels not in scan) replace them with 0 for the nifti
mean_vox_corr_wb_diff(isnan(mean_vox_corr_wb_diff)) = 0;

%% permutation testing apparently
n_permutations = 10000;
tic;
for it = 1:n_permutations
    time_elapsed = seconds(toc);
    time_per_perm = time_elapsed/it;
    time_elapsed.Format = 'dd:hh:mm:ss';
    fprintf('Permutations: %d/%d. Time per perm: %s. Total time elapsed: %s\n', it, n_permutations, time_per_perm, time_elapsed)

    % Per Phil 2024-03-04: The test statistic is once again the difference in correlations 
    % between ring-expand and the avg of the other 3
    % so for the permutations, return to permuting the sign of the difference
    % across xval folds
    % and now bc it's a contrast between stim types, no longer need to loop over stim types :3

    % generate the shuffler
    flipper=randi(2,1,n_valid_subs);
    % bc randi only lets you randomize positive ints. this cheat gets you +1s and -1s
    flipper(flipper==2)=-1;

    % actually shuffle the data
    for v=1:size(vox_corr_wb_diff,2)
        flipped_vox_corr_wb_diff(:,v)=vox_corr_wb_diff(:,v).*flipper';
    end

    % across the xval folds, collect the permuted means
    n_vox_wb_dist(it, :) = mean(flipped_vox_corr_wb_diff, "omitnan");
end

for v=1:size(mean_vox_corr_wb_diff, 2)
    phat_vox_wb(v) = 1-(sum(abs(mean_vox_corr_wb_diff(v))>abs(n_vox_wb_dist(:,v))))/(n_permutations+1);
end

% replace NaN voxels with p = 1 here
phat_vox_wb(isnan(phat_vox_wb)) = 1;

%% make statmap from permuted p-values
% for completion's sake, output one statmap per stim-specific-trained model
% even though the main one we care about is the expansion model
% in coherence with ms fig 1F, we don't need to do contrasts for this. just expansion

% it's not perfect, but we can grab the correct parameters for the fmri statmaps
% from the last fmri data that was read in
statmap = data;
pvalmap = data;
% flip all the removed voxels to not be removed
% bc all the NaNs in the stats/pvals were turned to 0 but that gives them values in the nifti
statmap.removed_voxels(statmap.removed_voxels) = false;
pvalmap.removed_voxels(pvalmap.removed_voxels) = false;

statmap.dat = mean_vox_corr_wb_diff(:)';
% in previous of Phil's code, the p-values were neg-log-scaled for visualization. Let us continue to do this
% this works with NaNs as 1 bc log10(1) = 0 yay
pvalmap.dat = -1*log10(phat_vox_wb(:));
statmap.fullpath = fullfile(studyforrest_dir, sprintf(out_fstring, 'stat'));
pvalmap.fullpath = fullfile(studyforrest_dir, sprintf(out_fstring, 'pval'));
write(statmap, 'overwrite');
write(pvalmap, 'overwrite');

