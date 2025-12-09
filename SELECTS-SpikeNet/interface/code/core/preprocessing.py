# core/preprocessing.py
import os
import traceback
import matlab.engine

def call_matlab_preprocess(file_path, fs=500, save_dir=r"D:\bai\interface\data\set"):
    channel_names = ['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6']
    
    try:
        file_name = os.path.basename(file_path)
        subject_id_mat = os.path.splitext(file_name)[0]
        set_filename_mat = subject_id_mat + '_raw_preprocessed.set'
        full_set_path = os.path.join(save_dir, set_filename_mat)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        eng = matlab.engine.start_matlab()
        eeglab_path = r"D:\MATLAB\R2024a\toolbox\eeglab"
        eng.addpath(eng.fullfile(eeglab_path), nargout=0)
        eng.addpath(eng.fullfile(eeglab_path, 'functions'), nargout=0)
        eng.addpath(eng.fullfile(eeglab_path, 'functions', 'guifunc'), nargout=0)
        eng.addpath(eng.fullfile(eeglab_path, 'functions', 'sigprocfunc'), nargout=0)
        
        chan_cell = eng.cell(len(channel_names), 1)
        for i, name in enumerate(channel_names):
            eng.workspace['name'] = name
            eng.eval(f'chan_cell{{{i+1}}} = name;', nargout=0)
        
        matlab_code = f"""
                if ~exist('{save_dir}', 'dir')
                    mkdir('{save_dir}');
                end

                % Load raw data (compatible with .set format, txt preprocessed to .set)
                EEG = pop_loadset('filename', '{os.path.basename(file_path)}', 'filepath', '{os.path.dirname(file_path)}');
                raw_data = EEG.data;

                % Fill NaN
                for i = 1:size(raw_data, 2)
                    nan_idx = isnan(raw_data(:, i));
                    if any(nan_idx)
                        raw_data(nan_idx, i) = mean(raw_data(~nan_idx, i), 'omitnan');
                    end
                end

                % New: Average re-reference (subtract average of all channels from each channel)
                avg_ref = mean(raw_data, 1);  % Calculate average of all channels (time point dimension)
                raw_data = raw_data - avg_ref;  % Apply average re-reference

                % 50Hz notch filter
                EEG_tmp = eeg_emptyset();
                EEG_tmp.data = raw_data;
                EEG_tmp.srate = {fs};
                EEG_tmp.nbchan = size(EEG_tmp.data, 1);
                EEG_tmp.pnts = size(EEG_tmp.data, 2);
                EEG_tmp.trials = 1;
                EEG_tmp = pop_eegfiltnew(EEG_tmp, 47, 53, 3300, 1);

                % 3-70Hz bandpass filter
                EEG_tmp = pop_eegfiltnew(EEG_tmp, 3, 70, [], 0);

                % Build EEG structure and save
                EEG = eeg_emptyset();
                EEG.data = EEG_tmp.data;
                EEG.srate = {fs};
                EEG.nbchan = size(EEG.data, 1);
                EEG.pnts = size(EEG.data, 2);
                EEG.trials = 1;
                EEG.times = (0:EEG.pnts - 1) / {fs};
                EEG.setname = '{subject_id_mat}';
                EEG.filename = '{set_filename_mat}';
                EEG.filepath = '{save_dir}';

                % Set channel names
                if length(chan_cell) ~= EEG.nbchan
                    warning('Number of channel names does not match');
                else
                    for ch = 1:EEG.nbchan
                        EEG.chanlocs(ch).labels = chan_cell{{ch}};
                        EEG.chanlocs(ch).type = 'EEG';
                    end
                end

                EEG = eeg_checkset(EEG);
                pop_saveset(EEG, 'filename', '{set_filename_mat}', 'filepath', '{save_dir}');
            """
        eng.eval(matlab_code, nargout=0)
        eng.quit()
        return full_set_path
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"MATLAB preprocessing failed: {str(e)}")