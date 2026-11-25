
"""Spike-wave and channel-level spatial metrics."""
import numpy as np

def calculate_swi(spike_times, total_duration):
    # 修复：先扁平化spike_times（处理可能混入的列表），再过滤无效值
    # 步骤1：扁平化数据（如果有嵌套列表，提取所有数值）
    flat_spike_times = []
    for t in spike_times:
        if isinstance(t, list):  # 如果元素是列表，展开
            flat_spike_times.extend([x for x in t if isinstance(x, (int, float))])
        elif isinstance(t, (int, float)):  # 如果是数值，直接添加
            flat_spike_times.append(t)

    # 步骤2：过滤范围外的时间（确保只保留0到total_duration之间的数值）
    valid_spike_times = [t for t in flat_spike_times if 0 <= t < total_duration]

    if not valid_spike_times or total_duration <= 0:
        return 0.0, 0, 0

    # 计算含棘波的秒数（取整数秒，去重）
    total_seconds = int(np.ceil(total_duration))
    spike_seconds = len(set([int(t) for t in valid_spike_times]))

    # 计算SWI值
    swi = (spike_seconds / total_seconds) * 100 if total_seconds > 0 else 0
    return round(swi, 2), spike_seconds, total_seconds


def calculate_high_pred1_ratios(raw_data, window_center_times, window_pred_map,
                               channel_names, target_channels, std_coeff=None):  # 移除原std_coeff默认值
    """
    计算各通道中"高值窗口且预测为1"的数量及比例
    新逻辑：高值窗口阈值 = 所有预测为1的窗口的最大值的均值
    """
    channel_ratios = {}
    channel_stats = {}

    for ch_idx, ch_name in enumerate(channel_names):
        if ch_name not in target_channels:
            continue

        # 1. 计算每个窗口的最大值，并区分预测为1和0的窗口
        win_max_list = []          # 所有窗口的最大值
        pred1_win_max_list = []    # 仅预测为1的窗口的最大值
        win_time_list = []

        for win_time in window_center_times:
            # 计算窗口在原始数据中的索引（确保不越界）
            total_time = raw_data.shape[1] / 500  # 总时长（秒）
            win_start_idx = int(round(win_time * 500)) - int(0.1 * 500)  # 0.1秒对应50采样点
            win_end_idx = win_start_idx + int(0.2 * 500)  # 窗口总长度0.2秒
            win_start_idx = max(0, win_start_idx)
            win_end_idx = min(raw_data.shape[1], win_end_idx)

            # 窗口内最大值
            win_max = np.max(raw_data[ch_idx, win_start_idx:win_end_idx])
            win_max_list.append(win_max)
            win_time_list.append(win_time)

            # 如果窗口预测为1，记录其最大值
            if window_pred_map.get(win_time, 0) == 1:
                pred1_win_max_list.append(win_max)

        total_windows = len(win_max_list)
        if total_windows == 0:
            channel_ratios[ch_name] = 0.0
            channel_stats[ch_name] = {
                'total_windows': 0,
                'high_windows': 0,
                'high_pred1_windows': 0,
                'ratio': 0.0,
                'high_threshold': 0.0  # 新增：记录阈值
            }
            continue

        # 2. 新逻辑：高值阈值 = 所有预测为1的窗口的最大值的均值
        if len(pred1_win_max_list) == 0:
            # 如果没有预测为1的窗口，阈值设为0（避免除零错误）
            high_threshold = 0.0
        else:
            # 核心修改：阈值 = 预测为1的窗口的最大值的均值
            high_threshold = np.mean(pred1_win_max_list)

        # 3. 统计"高值且预测为1"的窗口
        high_windows = 0  # 高值窗口总数（无论预测结果）
        high_pred1_windows = 0  # 高值且预测为1的窗口数

        for win_time, win_max in zip(win_time_list, win_max_list):
            # 判断是否为高值窗口（大于等于阈值）
            is_high = (win_max >= high_threshold)
            if not is_high:
                continue
            high_windows += 1

            # 判断该窗口是否预测为1
            if window_pred_map.get(win_time, 0) == 1:
                high_pred1_windows += 1

        # 4. 计算比例（高值且预测为1的窗口数 / 总窗口数）
        ratio = high_pred1_windows / total_windows if total_windows > 0 else 0.0
        channel_ratios[ch_name] = ratio
        channel_stats[ch_name] = {
            'total_windows': total_windows,
            'high_windows': high_windows,
            'high_pred1_windows': high_pred1_windows,
            'ratio': ratio,
            'high_threshold': high_threshold  # 新增：记录当前通道的阈值
        }

        # 打印通道统计结果（新增阈值信息）
        print(f"通道 {ch_name} 统计：")
        print(f"  总窗口数：{total_windows}")
        print(f"  预测为1的窗口数：{len(pred1_win_max_list)}")
        print(f"  高值阈值（预测为1的窗口最大值的均值）：{high_threshold:.4f}")
        print(f"  高值窗口数：{high_windows}")
        print(f"  高值且预测为1的窗口数：{high_pred1_windows}")
        print(f"  比例：{ratio:.2%}\n")

    return channel_ratios, channel_stats
