
"""Channel-level spatial-distribution visualization."""
import matplotlib.pyplot as plt
import mne
import numpy as np

def plot_brain_with_electrodes(raw_data, channel_names, target_channels, channel_counts, sfreq=500):
    """将电极定位（名称+位置）直接叠加在脑地形图上"""
    # 提取目标通道数据
    values = []
    ch_names_filtered = []
    for ch in target_channels:
        if ch in channel_names and ch in channel_counts:
            values.append(channel_counts[ch]['true_ratio'])
            ch_names_filtered.append(ch)

    if not ch_names_filtered:
        return plt.figure()

    # 获取标准10-20电极位置
    montage = mne.channels.make_standard_montage('standard_1020')
    pos_dict = montage.get_positions()['ch_pos']

    # 筛选有效通道及其坐标
    valid_chs = [ch for ch in ch_names_filtered if ch in pos_dict]
    if not valid_chs:
        return plt.figure()

    pos_array = np.array([pos_dict[ch] for ch in valid_chs])  # 保留x,y,z坐标
    values_norm = np.array([values[ch_names_filtered.index(ch)] for ch in valid_chs])
    if np.max(values_norm) > 0:
        values_norm = values_norm / np.max(values_norm)  # 归一化用于颜色映射
    else:
        values_norm = np.zeros_like(values_norm)

    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 10))

    # 绘制脑地形图（底图）
    im, _ = mne.viz.plot_topomap(
        values_norm,
        pos=pos_array[:, :2],  # 仅用x,y坐标
        axes=ax,
        cmap='jet',
        show=False,
        sensors=False,  # 不显示默认电极点（后续手动绘制）
        sphere=0.12  # 头部球体半径
    )
    ax.set_title('Channel-Level Spatial Distribution', fontsize=24, pad=18)
    cbar = plt.colorbar(im, ax=ax, label='Normalized Rate', shrink=0.8)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Normalized Rate', fontsize=16)

    # 计算脑图显示范围，动态调整标注偏移
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    brain_radius = max(xlim[1] - xlim[0], ylim[1] - ylim[0]) / 2  # 脑图显示半径
    offset_ratio = 0.05  # 偏移比例（控制标注与电极点的距离）
    offset = brain_radius * offset_ratio

    # 叠加电极点和名称（核心：定位叠在地形图上）
    for ch_name, (x, y, _) in zip(valid_chs, pos_array):
        # 绘制电极点（黑色圆点，带白色填充）
        ax.plot(x, y, 'ko', markersize=8, markeredgewidth=2, markerfacecolor='white')

        # 计算电极相对中心的距离，动态调整标注位置
        dist_from_center = np.sqrt(x**2 + y**2) / brain_radius

        if dist_from_center > 0.7:  # 边缘电极：沿径向向外偏移
            dir_x = x / np.sqrt(x**2 + y**2) if (x**2 + y**2) > 0 else 0
            dir_y = y / np.sqrt(x**2 + y**2) if (x**2 + y**2) > 0 else 0
            text_x = x + dir_x * offset
            text_y = y + dir_y * offset
        else:  # 内部电极：按象限偏移
            if x >= 0 and y >= 0:  # 右上
                text_x, text_y = x + offset, y + offset
            elif x >= 0 and y < 0:  # 右下
                text_x, text_y = x + offset, y - offset
            elif x < 0 and y >= 0:  # 左上
                text_x, text_y = x - offset, y + offset
            else:  # 左下
                text_x, text_y = x - offset, y - offset

        # 自动对齐文本（确保朝向脑图外侧）
        ha = 'left' if text_x > x else 'right'
        va = 'bottom' if text_y > y else 'top'

        # 标注电极名称（带白色背景框，避免遮挡地形图）
        ax.text(
            text_x, text_y,
            ch_name,
            fontsize=14,
            fontweight='bold',
            horizontalalignment=ha,
            verticalalignment=va,
            bbox=dict(facecolor='white', edgecolor='gray', pad=2, alpha=0.9)
        )

    plt.tight_layout()
    return fig
