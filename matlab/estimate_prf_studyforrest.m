
%% setup

addpath(genpath('/home/data/eccolab/Code/analyzePRF'));

%% actually do the things
studyforrest_dir = '/home/data/eccolab/studyforrest-data-phase2';
this_tr_length = 2;
this_n_folds = 1;
% ATTEND TO WHAT THIS PATH IS
sc_data = load(fullfile(studyforrest_dir, 'DATA_bpf.mat'));
v1_data = load(fullfile(studyforrest_dir, 'V1_DATA_bpf.mat'));
sc_prf = estimate_prf(sc_data, this_tr_length, this_n_folds);
DATA = sc_prf;
save('/home/data/eccolab/studyforrest-data-phase2/pred_sc_prf_groupavg.mat', "DATA")
v1_prf = estimate_prf(v1_data, this_tr_length, this_n_folds);
DATA = v1_prf;
save('/home/data/eccolab/studyforrest-data-phase2/pred_v1_prf_groupavg.mat', "DATA")

%% giant wrapper function to estimate pRFs from fMRI data

function DATA = estimate_prf(fmriData, trLength, nFolds)
    % put the stimulus dimension first because we're going to do it a stim at a
    % time
    % comes in as: subj, voxel, time, condition
    % if fMRI data doesn't come in as cell arrays by run,
    % it needs to be 4D with timepoint as the last dimension
    fmriData = permute(fmriData.DATA, [4 1 2 3]);
    % drop the non-task TRs
    fmriData = fmriData(:, :, :, 3:82);

    % preallocate array for predicted timecourse
    % note that the subject dimension has length nFolds
    % since we're training on one concatenated meta-subject per fold
    fmriPred = zeros(size(fmriData, 1), nFolds, size(fmriData, 3), size(fmriData, 4));

    conditionNames = {'wedge_counter' 'wedge_clock' 'ring_contract' 'ring_expand'};
    for stimNum=1:size(fmriData, 1)

        %% read and downsample stimuli
        disp(append('Starting stimulus condition: ', conditionNames{stimNum}))

        disp('reading in stimulus video')
        vidObj = VideoReader(fullfile('/home/mthieu/Repos/emonet-py/ignore', 'stimuli', 'studyforrest_retinotopy', append(conditionNames{stimNum}, '.mp4')));

        vidFrames = uint8(zeros(vidObj.Height, vidObj.Width, vidObj.NumFrames));
        for frame = 1:vidObj.NumFrames
            vidFrames(:, :, frame) = im2gray(read(vidObj, frame));
        end

        vidFrames = vidFrames ~= 46;

        vidFramesDownsampled = uint8(zeros(vidObj.Height, vidObj.Width, ceil(vidObj.NumFrames / (trLength*vidObj.FrameRate))));
        for frame = 1:size(vidFramesDownsampled, 3)
            idxStart = (frame-1)*50 + 1;
            if frame == size(vidFramesDownsampled, 3)
                idxEnd = size(vidFrames, 3);
            else
                idxEnd = idxStart + 50;
            end
            vidFramesDownsampled(:, :, frame) = mean(vidFrames(:, :, idxStart:idxEnd), 3);
        end
        vidFramesDownsampled = double(vidFramesDownsampled);

        % need to repeat the stimuli to align with the "subj-runs" created below
        if nFolds > 1
            vidFramesTrain = cell(((nFolds-1) / nFolds) * size(fmriData, 2));
        else 
            vidFramesTrain = cell(size(fmriData, 2));
        end

        for i=1:length(vidFramesTrain)
            vidFramesTrain{i} = vidFramesDownsampled;
        end
        vidFramesTrain = cat(3, vidFramesTrain{:});

        for fold=1:nFolds
            disp(append("starting fold ", string(fold)))
            %% drop the held-out test subjs

            trainSubjsLogical = true(size(fmriData, 2), 1);
            % should flexibly work with any number of xval folds...
            % EXCEPT 1 (no held-out test subjs)
            % because x:x:x includes everyone in testing when nFolds = 1
            if nFolds > 1
                testSubjs = fold:nFolds:size(fmriData, 2);   
                trainSubjsLogical(testSubjs) = false;
            end
            fmriDataTrain = fmriData(:, trainSubjsLogical, :, :);

            %% analyze PRFs hopefully
            % it's cognitively easier to do this inside the loop 
            % after squeezing out the condition dimension
            thisFmriData = squeeze(fmriDataTrain(stimNum, :, :, :));
            % put time and then subj together to create "subj-runs"
            thisFmriData = permute(thisFmriData, [2 3 1]);
            thisFmriData = reshape(thisFmriData, size(thisFmriData, 1), []);

            disp('analyzing pRFs')
            results = analyzePRF( ...
                vidFramesTrain, ...
                thisFmriData, ...
                trLength, ...
                struct('seedmode',-2,'display','off') ...
            );

            % note that this turns parallel pool on, we will LEAVE it on

            %% generate predicted timecourses
            % Kendrick's setup stuff (v slightly modified)
            % Define some variables
            res = size(vidFramesDownsampled, 1, 2);     % row x column resolution of the stimuli
            resmx = max(res);                           % maximum resolution (along any dimension)
            hrf = results.options.hrf;                  % HRF that was used in the model
            degs = results.options.maxpolydeg;          % vector of maximum polynomial degrees used in the model

            % Pre-compute cache for faster execution
            [d,xx,yy] = makegaussian2d(resmx,2,2,2,2);

            % Define the model function.  This function takes parameters and stimuli as input and
            % returns a predicted time-series as output.  Specifically, the variable <pp> is a vector
            % of parameter values (1 x 5) and the variable <dd> is a matrix with the stimuli (frames x pixels).
            % Although it looks complex, what the function does is pretty straightforward: construct a
            % 2D Gaussian, crop it to <res>, compute the dot-product between the stimuli and the
            % Gaussian, raise the result to an exponent, and then convolve the result with the HRF,
            % taking care to not bleed over run boundaries.
            modelfun = @(pp,dd) conv2run(posrect(pp(4)) * (dd*[vflatten(placematrix(zeros(res),makegaussian2d(resmx,pp(1),pp(2),abs(pp(3)),abs(pp(3)),xx,yy,0,0) / (2*pi*abs(pp(3))^2))); 0]) .^ posrect(pp(5)),hrf,dd(:,prod(res)+1));

            % Reshape the stimuli to frames x pixels
            % (use non-repeated stimuli bc test is projected separately per subj)
            vidFrames2d = squish(vidFramesDownsampled,2)';
            % Kendrick's example script adds a dummy column for run breaks
            % we only have one run here but I suspect modelfun is expecting that extra column so, fine.
            vidFrames2d = [vidFrames2d ones(size(vidFrames2d,1),1)];


            % Construct projection matrix that fits and removes the polynomials.
            polymatrix = projectionmatrix(constructpolynomialmatrix(size(fmriData,4),0:degs));

            % Output only the model predicted polynomial timecourse thingy
            % since we are pushing this over to R for model comparison anyway
            
            % Iterate over every OTHER dimension in the fMRI data
            % I think nest the for loops from the outside in
            % so that the repeating goes in the same order that matlab repeats array indices
            disp('generating predicted timecourses')
            % for safety, only run this one on parfor
            % bc the modelfun call seems to be the one that takes 5ever
            parfor voxel=1:size(fmriPred, 3)
                fmriPred(stimNum, fold, voxel, :) = polymatrix*modelfun(results.params(1,:,voxel),vidFrames2d);
            end
        end
    end

    %% save and get the fuck out of here
    disp('finished :3')
    % reorganize dims back into the dim order of Phil's fMRI data output
    DATA = permute(fmriPred, [2 3 4 1]);
end
