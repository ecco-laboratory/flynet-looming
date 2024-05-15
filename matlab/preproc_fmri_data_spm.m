%-----------------------------------------------------------------------
% Job saved on 15-Dec-2022 16:32:37 by cfg_util (rev $Rev: 7345 $)
% spm SPM - SPM12 (7771)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------
%% SPM setup
% paths must be specified from the directory in which the file is _saved_
spm_dir = '../ignore/libraries/spm12';
addpath(spm_dir);
% must wake up spm shit manually bc calling through cmd line
spm('defaults','fmri');
spm_jobman('initcfg');

% this script reads in raw studyforrest data
% so I think it's kosher to set this path in this script instead of targets-tracking
studyforrest_dir = '../ignore/datasets/studyforrest-data-phase2';

%% VARIABLES THAT MUST BE DEFINED BEFORE THE SCRIPT IS SOURCED
% paths_nifti_[ccw, clw, con, exp]: 4 cell arrays with the `,n` TR digits appended to each internal filename
% out_prefix: set in the targets file so that it's forcibly consistent

%% SPM MUMBO JUMBO
% to realign across sessions together, the data from all 4 retinotopy stimuli for each subject must be called together
% accordingly, 
matlabbatch{1}.spm.spatial.realign.estimate.data = {paths_nifti_ccw; paths_nifti_clw; paths_nifti_con; paths_nifti_exp};
matlabbatch{1}.spm.spatial.realign.estimate.eoptions.quality = 0.9;
matlabbatch{1}.spm.spatial.realign.estimate.eoptions.sep = 4;
matlabbatch{1}.spm.spatial.realign.estimate.eoptions.fwhm = 5;
matlabbatch{1}.spm.spatial.realign.estimate.eoptions.rtm = 1;
matlabbatch{1}.spm.spatial.realign.estimate.eoptions.interp = 2;
matlabbatch{1}.spm.spatial.realign.estimate.eoptions.wrap = [0 0 0];
matlabbatch{1}.spm.spatial.realign.estimate.eoptions.weight = '';
% this should get the first wedge counter and return it as a 1x1 cell array
matlabbatch{2}.spm.tools.oldnorm.estwrite.subj.source = paths_nifti_ccw(1);
matlabbatch{2}.spm.tools.oldnorm.estwrite.subj.wtsrc = '';
matlabbatch{2}.spm.tools.oldnorm.estwrite.subj.resample(1) = cfg_dep('Realign: Estimate: Realigned Images (Sess 1)', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','sess', '()',{1}, '.','cfiles'));
matlabbatch{2}.spm.tools.oldnorm.estwrite.subj.resample(2) = cfg_dep('Realign: Estimate: Realigned Images (Sess 2)', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','sess', '()',{2}, '.','cfiles'));
matlabbatch{2}.spm.tools.oldnorm.estwrite.subj.resample(3) = cfg_dep('Realign: Estimate: Realigned Images (Sess 3)', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','sess', '()',{3}, '.','cfiles'));
matlabbatch{2}.spm.tools.oldnorm.estwrite.subj.resample(4) = cfg_dep('Realign: Estimate: Realigned Images (Sess 4)', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','sess', '()',{4}, '.','cfiles'));
matlabbatch{2}.spm.tools.oldnorm.estwrite.eoptions.template = {fullfile(spm_dir, 'toolbox/OldNorm/EPI.nii,1')};
matlabbatch{2}.spm.tools.oldnorm.estwrite.eoptions.weight = '';
matlabbatch{2}.spm.tools.oldnorm.estwrite.eoptions.smosrc = 8;
matlabbatch{2}.spm.tools.oldnorm.estwrite.eoptions.smoref = 0;
matlabbatch{2}.spm.tools.oldnorm.estwrite.eoptions.regtype = 'mni';
matlabbatch{2}.spm.tools.oldnorm.estwrite.eoptions.cutoff = 25;
matlabbatch{2}.spm.tools.oldnorm.estwrite.eoptions.nits = 16;
matlabbatch{2}.spm.tools.oldnorm.estwrite.eoptions.reg = 1;
matlabbatch{2}.spm.tools.oldnorm.estwrite.roptions.preserve = 0;
matlabbatch{2}.spm.tools.oldnorm.estwrite.roptions.bb = [-78 -112 -70
                                                         78 76 85];
matlabbatch{2}.spm.tools.oldnorm.estwrite.roptions.vox = [2 2 2];
matlabbatch{2}.spm.tools.oldnorm.estwrite.roptions.interp = 1;
matlabbatch{2}.spm.tools.oldnorm.estwrite.roptions.wrap = [0 0 0];
matlabbatch{2}.spm.tools.oldnorm.estwrite.roptions.prefix = out_prefix;

%% execute the job!
spm_jobman('run', matlabbatch);
