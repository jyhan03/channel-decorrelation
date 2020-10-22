% spatialize_wsj0_mix.m
%
% Create spatialized versions of the wsj0-mix dataset mixtures
% 
% This script assumes that the wsj0-mix dataset has already been created.
%
% The information necessary to generate the RIRs is assumed to have already
% been generated, for example using sample_RIRs.m, in a file rir_info.mat.
% This script:
%    - Generates the corresponding RIRs assuming anechoic and reverberant
%      conditions;
%    - Filters each source signal with the corresponding RIR;
%    - Scales the anechoic source images (i.e., direct sound) at the 
%      microphone to match the SNR of the original mixture. 
%    - Scales the reverberant signals with the same factor
%    - Rescale all signals so that maximum sample value overall is 0.9.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Copyright (C) 2017-2018 Mitsubishi Electric Research Labs 
%                     (Zhong-Qiu Wang, Jonathan Le Roux, John R. Hershey)
%   Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function spatialize_wsj0_mix(num_speakers,min_or_max,fs,start_ind,stop_ind,useparcluster_with_ind,generate_rirs)
    % add path to RIR generator
    addpath('/Share/hjy/data/origin_data/wsj_wav/tools/RIR-Generator/'); 
    
    data_in_root  = '/Share/hjy/data/SE/simulation/'; % directory of the original wsj0-mix data
    rir_root      = '/Share/hjy/data/SE/simulation/'; % directory to write the generated RIRs into
    data_out_root = '/Share/hjy/data/SE/';            % directory to write the spatialized mixtures into
    
    if ~exist('num_speakers','var')
        num_speakers = 2;
    end
    if ~exist('min_or_max','var')
        min_or_max = 'min';
    end
    if ~exist('fs','var')
        fs = 8000;
    end
    if ~exist('useparcluster_with_ind','var')
        useparcluster_with_ind = 0;
    end
    if ~exist('generate_rirs','var')
        generate_rirs= 1;
    end
    
    fs_str = [num2str(fs/1000),'k'];
    rir_dir = [rir_root, '/RIRs_', fs_str, '/'];
    if ~isdir(rir_dir)
        mkdir(rir_dir);
    end

    tmp = load('rir_info.mat');
    INFO = tmp.INFO;

    % Determine whether to use parallel processing toolbox or not
    useparcluster = 0;
    if exist('parcluster','file')
        useparcluster = 1;
    end
    
    NUM_UTT_IN_WSJ_MIX = 28000;
    if ~exist('start_ind','var')
        start_ind=1;
        stop_ind = NUM_UTT_IN_WSJ_MIX;
    elseif ~exist('stop_ind','var')
        error('If a starting index is specified, a stopping index should be too');
    else % typically, indices are used to parallelize outside matlab, but one can force use of PP toolbox
        useparcluster = useparcluster_with_ind; 
        stop_ind = min(stop_ind,NUM_UTT_IN_WSJ_MIX);
    end
    if generate_rirs
        generate_RIRs(rir_dir, start_ind, stop_ind, fs, INFO, useparcluster);
    end

    spatialize_mix(rir_dir, start_ind, stop_ind, fs, data_in_root, data_out_root, num_speakers, min_or_max, useparcluster);

end

function spatialize_mix(rir_dir, start_ind, stop_ind, fs, data_in_root, data_out_root,num_speakers,min_or_max, useparcluster)
    % Deal with the wavread/audioread and wavwrite/audiowrite annoyance
    useaudioread = 0;
    if exist('audioread','file')
        useaudioread = 1;
    end
        
    fs_str = [num2str(fs/1000),'k'];

    datasets = {'tr', 'cv', 'tt'};
    rir_offsets = [0,20000,25000];
    source_strs = cell(num_speakers+1,1);
    for ss=1:num_speakers
        source_strs{ss} = sprintf('s%d/',ss);
    end
    source_strs{num_speakers+1} = 'mix/';

    if useparcluster
        c = parcluster('local');
        c.NumWorkers = 22;
        parpool(c, c.NumWorkers);
    else
        c = struct;
        c.NumWorkers = 0;
    end

    for dd = 1 : length(datasets)
        dataset = datasets{dd};
        rir_offset = rir_offsets(dd);

        flist = importdata(['wsj0-',num2str(num_speakers),'mix_',dataset,'.flist']);        
        fprintf('Processing dataset %s with %d mixtures\n',dataset,length(flist));
        anechoic_root_dir = [data_out_root,'/',num2str(num_speakers),'speakers_anechoic/wav',fs_str,'/',min_or_max,'/',dataset,'/'];
        reverb_root_dir = [data_out_root,'/',num2str(num_speakers),'speakers_reverb/wav',fs_str,'/',min_or_max,'/',dataset,'/'];
        for ss=1:(num_speakers+1)
            if ~isdir([anechoic_root_dir source_strs{ss}]) %#ok<*ISDIR>
                mkdir([anechoic_root_dir source_strs{ss}]);
            end
            if ~isdir([reverb_root_dir source_strs{ss}])
                mkdir([reverb_root_dir source_strs{ss}]);
            end
        end

        parfor (fInd = 1 : length(flist),c.NumWorkers)
        %for fInd = 1 : length(flist)
            
            if (start_ind <= rir_offset+fInd) && (rir_offset+fInd <= stop_ind)
                % only process thos RIRs that are in the range

                filename = flist{fInd};

                rir_file = [rir_dir, sprintf('rir_%05d.mat',rir_offset+fInd)];
                fprintf('using RIR %s, spatializing %s...', rir_file, filename);
                info_ = load(rir_file);
                fprintf('T60=%f...', info_.T60);

                source_dir = [data_in_root,'/',num2str(num_speakers),'speakers/wav',fs_str,'/',min_or_max,'/',dataset,'/'];

                % cell array containing raw, anechoic, reverb signals for each source and the mixture:
                % raw_1, ..., raw_S, []
                % anechoic_1, ..., anechoic_S, anechoic_mix
                % reverb_1, ..., reverb_S, reverb_mix
                signals = cell(3,num_speakers+1);

                % Read the original sources in
                for ss=1:num_speakers
                    if useaudioread
                        signals{1,ss} = audioread([source_dir,'s',num2str(ss),'/',filename]);
                    else
                        signals{1,ss} = wavread([source_dir,'s',num2str(ss),'/',filename]); %#ok<*DWVRD>
                    end
                end

                % Apply RIRs
                % NOTE: the first segment of anechoic rir is matched with that of reverb rir
                for ss=1:num_speakers
                    h_anechoic = info_.(['h_anechoic_',num2str(ss)]);
                    h_reverb   = info_.(['h_reverb_',num2str(ss)]);
                    signals{2,ss} = synthesis_reverberation(h_anechoic, signals{1,ss});
                    signals{3,ss} = synthesis_reverberation(h_reverb,   signals{1,ss});
                end

                % Get the original SNR between the first (and opt. second) and the last speaker
                original_snr = snr(signals(1,:));
                fprintf('snr=%.3f...', original_snr);

                % Match the SNR of anechoic signals to original SNR,
                % and rescale the reverb signals with the same factor
                signals = scale_to_snr(signals, original_snr);

                % Build the anechoic and reverb mixtures
                signals{2,num_speakers+1} = sum(cat(3,signals{2,1:num_speakers}),3); 
                signals{3,num_speakers+1} = sum(cat(3,signals{3,1:num_speakers}),3); 

                % Rescale everyone so that max absolute value is 0.9
                signals = scale_to_max_sample_value(signals, 0.9);

                % Write the wav files
                for ss=1:(num_speakers+1)
                    save_wav([anechoic_root_dir,source_strs{ss},filename], signals{2,ss}, fs, useaudioread); 
                    save_wav([reverb_root_dir,source_strs{ss},filename], signals{3,ss}, fs, useaudioread);
                end

                fprintf('done\n');
            end

        end
    end
    if useparcluster
        delete(gcp);
    end
end

function save_wav(filepath, signal, fs, useaudioread)
    if useaudioread
        signal = int16(round((2^15)*signal));
        audiowrite(filepath, signal, fs);
    else
        wavwrite(signal, fs, filepath); %#ok<*DWVWR>
    end
end


function ret = snr(sigs)
    num_speakers = size(sigs,2)-1;
    ret = zeros(num_speakers-1,1);
    sref = sigs{1,num_speakers};
    for ss = 1:(num_speakers-1)
        s = sigs{1,ss};
        ret(ss) = 10*log10(sum(s(:).^2)/sum(sref(:).^2));
    end
end

function signals = scale_to_snr(signals, original_snr)
    %
    % use the first four channels only to compute the scaling_factor, as initially only four mics were simulated
    %
    num_speakers = size(signals,2)-1;
    anechoic_sref = signals{2,num_speakers};
    for ss = 1:(num_speakers-1)        
        anechoic_s = signals{2,ss};
        scaling_factor  = sqrt(10^(original_snr(ss)/10)*sum(sum(anechoic_sref(:,1:4).^2))/sum(sum(anechoic_s(:,1:4).^2)));
        signals{2,ss} = signals{2,ss} * scaling_factor;
        signals{3,ss} = signals{3,ss} * scaling_factor;
    end
end

function signals = scale_to_max_sample_value(signals,max_sample_value)
    tmp = cell2mat(cellfun(@max,(cellfun(@abs,signals(2:3,:),'un',0)),'un',0));
    max_ = max(tmp(:));
    signals = cellfun(@(x) x * max_sample_value / max_, signals,'un',0);
end

function reverbsyn = synthesis_reverberation(rir, speech)
    rir = rir'; %rir should be filter_len*M
    reverbsyn = zeros(length(speech),size(rir,2));
    for n = 1:size(rir,2)
        lreverb = conv(speech,rir(:,n));
        reverbsyn(:,n) = lreverb(1:length(speech));
    end
end

function generate_RIRs(rir_dir, start_ind, stop_ind, fs, INFO, useparcluster)
   
    if useparcluster
        c = parcluster('local');
        c.NumWorkers = 22;
        parpool(c, c.NumWorkers);
    else
        c.NumWorkers = 0;
    end

    parfor (kk = start_ind : stop_ind,c.NumWorkers)

        fprintf('processing %d of [%d-%d], ', kk, start_ind, stop_ind);

        mtype   = INFO(kk).mtype; % Type of microphone
        order   = INFO(kk).order; % -1 equals maximum reflection order!
        dim     = INFO(kk).dim;   % Room dimension
        orientation = INFO(kk).orientation; % Microphone orientation (rad), doesn't matter for omnidirectional microphones
        hp_filter   = INFO(kk).hp_filter;   % Enable high-pass filter
        sound_speed = 340;                  % Sound velocity (m/s)

        room_dimension = INFO(kk).room_dimension; % Room dimensions
        mic_pos     = INFO(kk).mic_pos;
        spk_pos     = INFO(kk).spk_pos;


        %
        % for reverb
        %
        T60             = INFO(kk).T60;  % Reverberation time (s)
        filter_len      = fs*(T60+0.1);
        fprintf('T60=%f\n', T60);
        h_reverb_1      = rir_generator(sound_speed, fs, mic_pos, spk_pos(1,:), room_dimension, T60, filter_len, mtype, order, dim, orientation, hp_filter);
        h_reverb_2      = rir_generator(sound_speed, fs, mic_pos, spk_pos(2,:), room_dimension, T60, filter_len, mtype, order, dim, orientation, hp_filter);
        h_reverb_3      = rir_generator(sound_speed, fs, mic_pos, spk_pos(3,:), room_dimension, T60, filter_len, mtype, order, dim, orientation, hp_filter);

        %
        % for anechoic
        %
        filter_len      = fs*0.1;
        h_anechoic_1    = rir_generator(sound_speed, fs, mic_pos, spk_pos(1,:), room_dimension, 0.0, filter_len, mtype, order, dim, orientation, hp_filter);
        h_anechoic_2    = rir_generator(sound_speed, fs, mic_pos, spk_pos(2,:), room_dimension, 0.0, filter_len, mtype, order, dim, orientation, hp_filter);
        h_anechoic_3    = rir_generator(sound_speed, fs, mic_pos, spk_pos(3,:), room_dimension, 0.0, filter_len, mtype, order, dim, orientation, hp_filter);

        save_rir(rir_dir, kk, h_reverb_1, h_reverb_2, h_reverb_3, h_anechoic_1, h_anechoic_2, h_anechoic_3, fs, mic_pos, spk_pos, room_dimension, T60, mtype, order, dim, orientation, hp_filter);
            
    end

    if useparcluster
        delete(gcp);
    end

end

function save_rir(rir_dir, kk, h_reverb_1, h_reverb_2, h_reverb_3, h_anechoic_1, h_anechoic_2, h_anechoic_3, fs, mic_pos, spk_pos, room_dimension, T60, mtype, order, dim, orientation, hp_filter) %#ok<INUSD>
    save([rir_dir,'/rir_',sprintf('%05d',kk),'.mat'], 'h_reverb_1', 'h_reverb_2', 'h_reverb_3', 'h_anechoic_1', 'h_anechoic_2', 'h_anechoic_3', 'fs', 'mic_pos', 'spk_pos', 'room_dimension', 'T60', 'mtype', 'order', 'dim', 'orientation', 'hp_filter');
end
