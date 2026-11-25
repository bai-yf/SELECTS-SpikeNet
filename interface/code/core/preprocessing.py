
"""Reusable MATLAB/EEGLAB engine lifecycle and preprocessing pipeline."""
import os
import threading
import traceback


from config import EEGLAB_PATH, INTERFACE_DATA_DIR

_MATLAB_ENGINE = None
_MATLAB_ENGINE_PATHS_READY = False
_MATLAB_ENGINE_LOCK = threading.Lock()
_MATLAB_WARMUP_THREAD = None
_MATLAB_SHUTDOWN_REQUESTED = False

def get_matlab_engine():
    """Start MATLAB once per application session and reuse the engine."""
    import matlab.engine

    global _MATLAB_ENGINE, _MATLAB_ENGINE_PATHS_READY
    with _MATLAB_ENGINE_LOCK:
        if _MATLAB_SHUTDOWN_REQUESTED:
            raise RuntimeError("MATLAB startup was canceled because the application is closing.")

        if _MATLAB_ENGINE is None:
            started_engine = matlab.engine.start_matlab()
            if _MATLAB_SHUTDOWN_REQUESTED:
                started_engine.quit()
                raise RuntimeError("MATLAB startup was canceled because the application is closing.")
            _MATLAB_ENGINE = started_engine
            _MATLAB_ENGINE_PATHS_READY = False

        if not _MATLAB_ENGINE_PATHS_READY:
            eeglab_path = EEGLAB_PATH
            _MATLAB_ENGINE.addpath(_MATLAB_ENGINE.fullfile(eeglab_path), nargout=0)
            _MATLAB_ENGINE.addpath(_MATLAB_ENGINE.fullfile(eeglab_path, 'functions'), nargout=0)
            _MATLAB_ENGINE.addpath(_MATLAB_ENGINE.fullfile(eeglab_path, 'functions', 'guifunc'), nargout=0)
            _MATLAB_ENGINE.addpath(_MATLAB_ENGINE.fullfile(eeglab_path, 'functions', 'sigprocfunc'), nargout=0)
            _MATLAB_ENGINE_PATHS_READY = True
        return _MATLAB_ENGINE


def matlab_engine_is_ready():
    return _MATLAB_ENGINE is not None and _MATLAB_ENGINE_PATHS_READY


def warm_up_matlab_engine_async():
    """Start MATLAB/EEGLAB in the background while TXT data are imported."""
    global _MATLAB_WARMUP_THREAD
    if matlab_engine_is_ready():
        return
    if _MATLAB_WARMUP_THREAD is not None and _MATLAB_WARMUP_THREAD.is_alive():
        return

    def worker():
        try:
            get_matlab_engine()
        except Exception:
            if not _MATLAB_SHUTDOWN_REQUESTED:
                traceback.print_exc()

    _MATLAB_WARMUP_THREAD = threading.Thread(
        target=worker,
        name="matlab-engine-warmup",
        daemon=True,
    )
    _MATLAB_WARMUP_THREAD.start()


def call_matlab_preprocess(file_path, fs=500, save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(INTERFACE_DATA_DIR, "set")
    channel_names = ['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6']

    try:
        file_name = os.path.basename(file_path)
        subject_id_mat = os.path.splitext(file_name)[0]
        set_filename_mat = subject_id_mat + '_raw_preprocessed.set'
        full_set_path = os.path.join(save_dir, set_filename_mat)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        eng = get_matlab_engine()

        chan_cell = eng.cell(len(channel_names), 1)
        for i, name in enumerate(channel_names):
            eng.workspace['name'] = name
            eng.eval(f'chan_cell{{{i+1}}} = name;', nargout=0)

        matlab_code = f"""
                if ~exist('{save_dir}', 'dir')
                    mkdir('{save_dir}');
                end

                % 加载原始数据（兼容.set格式，txt预处理后已转为.set）
                EEG = pop_loadset('filename', '{os.path.basename(file_path)}', 'filepath', '{os.path.dirname(file_path)}');
                raw_data = EEG.data;

                % Vectorized NaN replacement. Each missing channel value is
                % replaced by the mean of the available channels at that time.
                nan_mask = isnan(raw_data);
                if any(nan_mask(:))
                    column_mean = mean(raw_data, 1, 'omitnan');
                    column_mean(isnan(column_mean)) = 0;
                    nan_linear = find(nan_mask);
                    [~, nan_columns] = ind2sub(size(raw_data), nan_linear);
                    raw_data(nan_linear) = reshape(column_mean(nan_columns), [], 1);
                end

                % 新增：平均重参考（将每个通道数据减去所有通道的平均值）
                avg_ref = mean(raw_data, 1);  % 计算所有通道的平均值（时间点维度）
                raw_data = raw_data - avg_ref;  % 应用平均重参考

                % 50Hz陷波滤波
                EEG_tmp = eeg_emptyset();
                EEG_tmp.data = raw_data;
                EEG_tmp.srate = {fs};
                EEG_tmp.nbchan = size(EEG_tmp.data, 1);
                EEG_tmp.pnts = size(EEG_tmp.data, 2);
                EEG_tmp.trials = 1;
                EEG_tmp = pop_eegfiltnew(EEG_tmp, 47, 53, 3300, 1);

                % 3-70Hz带通滤波
                EEG_tmp = pop_eegfiltnew(EEG_tmp, 3, 70, [], 0);

                % 构建EEG结构并保存
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

                % 设置通道名
                if length(chan_cell) ~= EEG.nbchan
                    warning('通道名称数量不匹配');
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
        return full_set_path
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"MATLAB preprocessing failed: {str(e)}")


def shutdown_matlab_engine():
    """Stop the shared MATLAB engine, if running, during application shutdown."""
    global _MATLAB_ENGINE, _MATLAB_ENGINE_PATHS_READY, _MATLAB_SHUTDOWN_REQUESTED
    _MATLAB_SHUTDOWN_REQUESTED = True
    with _MATLAB_ENGINE_LOCK:
        if _MATLAB_ENGINE is not None:
            try:
                _MATLAB_ENGINE.quit()
            except Exception:
                pass
            _MATLAB_ENGINE = None
            _MATLAB_ENGINE_PATHS_READY = False
