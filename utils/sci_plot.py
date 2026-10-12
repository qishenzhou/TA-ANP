import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Tuple, Optional
import numpy.typing as npt
from scipy import stats
import math
import os
import pandas as pd
import seaborn as sns
from scipy.stats import norm


def plot_multi_timeseries_shared_axes(
    nrows,
    ncols,
    subplot_data_mapping,  # List of tuples: [(row, col, data_list, {'title': '...'})]
    subplot_aspect_ratio=1.0,  # Width / Height ratio for each subplot
    figure_width=7.0,  # Default width in inches (approx. A4 usable width)
    default_xlabel="Time",
    default_ylabel="Value",
    # --- Styling Defaults ---
    dpi=300,
    linewidth=1,
    markersize=2.5,
    legend_loc='upper center',
    legend_bbox_to_anchor=(0.5,
                           -0.1),  # Anchor point close below figure center
    legend_ncol=None,  # Default: Calculate based on number of labels
    legend_fontsize=8,
    label_fontsize=8,
    subplot_title_fontsize=8,
    subplot_title_pad=2,  # Padding below subplot title (points)
    tick_fontsize=8,
    tick_pad=2,
    tick_length=2,
    num_xticks=3,
    label_pad=3,  # Padding for outer axis labels (increased slightly)
    use_grid=True,
    font_family=['Arial', 'Noto Sans SC'
                 ],  # Fallback to Noto Sans for Chinese characters
    save_path=None,
    file_format='png',
    subplot_hspace=0.25,
    axis_limit_padding=0.3,  # Relative padding for axis limits (3%)
    min_y_global=0,
    max_y_global=600,
):
    """
    Creates publication-quality multi-subplot time series plots with shared axes.
    (Final Version: Shared axes, compact layout, controlled labels/ticks, no main title)

    Features automatic figure height calculation, shared X and Y axes with
    synchronized limits and padding, targeted data plotting, mandatory subplot titles,
    compact layout, first column Y labels, last row X labels, outer tick labels only,
    and a single shared legend closely below the figure.

    Args:
        nrows (int): Number of rows in the subplot grid.
        ncols (int): Number of columns in the subplot grid.
        subplot_data_mapping (list): Maps data to subplots. Each element is a tuple:
                                     (row_index, col_index, data_list, options_dict).
                                     'data_list' contains {'x', 'y', 'label', optional styling}.
                                     'options_dict' *must* contain {'title': 'Subplot Title'}.
        subplot_aspect_ratio (float): Desired width/height ratio for each *individual* subplot.
        figure_width (float): The desired total width of the figure in inches.
        default_xlabel (str): Label for the shared x-axis (shown on bottom row).
        default_ylabel (str): Label for the shared y-axis (shown on left column).
        dpi (int): Figure resolution.
        colors (list, optional): Default color cycle.
        linestyles (list, optional): Default linestyle cycle.
        markers (list, optional): Default marker cycle.
        linewidth (float): Default line width.
        markersize (float): Default marker size.
        legend_loc (str): Anchor point location *within* the legend box.
        legend_bbox_to_anchor (tuple): Anchor point for the legend box relative to the figure.
        legend_ncol (int, optional): Number of columns for the shared legend. Auto-calculated if None.
        legend_fontsize (int): Font size for the shared legend.
        label_fontsize (int): Font size for outer axis labels.
        subplot_title_fontsize (int): Font size for individual subplot titles.
        subplot_title_pad (float): Padding below subplot titles in points.
        tick_fontsize (int): Font size for outer axis ticks.
        tick_pad (float): Padding for tick labels.
        tick_length (float): Length of tick marks.
        num_xticks (int): Number of x-axis ticks.
        label_pad (float): Padding for outer axis labels.
        use_grid (bool): Whether to display grid lines on subplots.
        font_family (str): Global font family.
        save_path (str, optional): Path to save the figure (without extension).
        file_format (str): Format for saving ('png', 'pdf', 'svg', etc.).
        axis_limit_padding (float): Relative padding added to min/max axis limits (e.g., 0.03 for 3%).

    Returns:
        tuple: (matplotlib.figure.Figure, np.ndarray of matplotlib.axes.Axes)
               The figure and the 2D array of axes objects.
    """
    plt.rcParams['font.family'] = font_family
    plt.rcParams['font.size'] = tick_fontsize

    # --- Calculate Figure Size ---
    subplot_width = figure_width / ncols
    subplot_height = subplot_width / subplot_aspect_ratio
    total_height_estimate = subplot_height * nrows

    # Estimate buffer needed for legend below
    legend_rows_estimate = 1
    unique_label_count = len(
        set(item['label'] for map_item in subplot_data_mapping
            for item in map_item[2] if 'label' in item))
    temp_legend_ncol = legend_ncol
    if temp_legend_ncol is None and unique_label_count > 0:
        max_per_row = math.floor(figure_width / (legend_fontsize / 10 * 1.5))
        temp_legend_ncol = min(unique_label_count, max(1, int(max_per_row)))
    if temp_legend_ncol is not None and temp_legend_ncol > 0 and unique_label_count > 0:
        legend_rows_estimate = math.ceil(unique_label_count / temp_legend_ncol)

    # Buffer for legend + small top/bottom margins
    height_buffer_legend = 0.2 + (
        legend_fontsize / 72 * 1.3 * legend_rows_estimate
    ) if unique_label_count > 0 else 0.05  # Tight multiplier
    total_height = total_height_estimate + height_buffer_legend

    # --- Create Subplots with Shared Axes ---
    fig, axes = plt.subplots(nrows,
                             ncols,
                             figsize=(figure_width, total_height),
                             dpi=dpi,
                             sharex=True,
                             sharey=True,
                             squeeze=False)

    # --- Setup Default Styles ---
    default_colors = plt.cm.tab10.colors
    default_linestyles = ['-', '--', '-.', ':']

    # --- Plot Data and Track Limits/Labels ---
    unique_labels = {}
    min_x_global, max_x_global = np.inf, -np.inf  # Track actual X if numeric
    min_x_idx_global, max_x_idx_global = 0, 0
    x_values_list_global = None
    is_x_string = False

    for mapping_item in subplot_data_mapping:
        if len(mapping_item) != 4:
            raise ValueError(
                "subplot_data_mapping items must be (row, col, data_list, options_dict)"
            )

        row, col, data_list, options_dict = mapping_item
        subplot_title = options_dict.get('title', None)
        if subplot_title is None:
            raise ValueError(
                f"Missing 'title' in options_dict for subplot ({row},{col})")

        if row >= nrows or col >= ncols:
            print(
                f"Warning: Subplot index ({row}, {col}) out of bounds. Skipping."
            )
            continue

        ax = axes[row, col]
        ax.set_title(subplot_title,
                     fontsize=subplot_title_fontsize,
                     pad=subplot_title_pad)

        if not data_list:
            ax.text(0.5,
                    0.5,
                    '(No data)',
                    ha='center',
                    va='center',
                    fontsize=tick_fontsize,
                    color='gray',
                    transform=ax.transAxes)
            continue

        # Determine x-axis type and store reference values from first valid series
        if x_values_list_global is None and data_list[0].get('x') is not None:
            _x_temp = data_list[0]['x']
            if hasattr(_x_temp, '__iter__') and not isinstance(
                    _x_temp, str) and len(_x_temp) > 0:
                x_values_list_global = list(_x_temp)
                if isinstance(x_values_list_global[0], str):
                    is_x_string = True
                    # Check if it's time format 'HH:MM'
                    if all(':' in str(x) for x in x_values_list_global):
                        # For time strings, we'll use minutes as numeric values
                        def time_to_minutes(time_str):
                            hours, minutes = map(int, time_str.split(':'))
                            return hours * 60 + minutes

                        time_minutes = [
                            time_to_minutes(x) for x in x_values_list_global
                        ]
                        min_x_global = min(time_minutes)
                        max_x_global = max(time_minutes)
                    else:
                        # Non-time strings, use index-based approach
                        max_x_idx_global = len(x_values_list_global) - 1
                else:  # Assume numeric X
                    try:
                        _x_numeric = np.array(x_values_list_global,
                                              dtype=float)
                        min_x_global = np.nanmin(_x_numeric)
                        max_x_global = np.nanmax(_x_numeric)
                    except ValueError:
                        print(
                            "Warning: Could not convert X data to numeric, treating as categorical index."
                        )
                        is_x_string = True
                        max_x_idx_global = len(x_values_list_global) - 1

        # --- Plotting loop ---
        for i, data in enumerate(data_list):
            x_raw = data['x']
            y = data['y']
            label = data.get('label', f'Series {row}-{col}-{i+1}')
            color = data.get('color', default_colors[i % len(default_colors)])
            linestyle = data.get(
                'linestyle', default_linestyles[i % len(default_linestyles)])
            marker = data.get('marker', 'None')
            lw = data.get('linewidth', linewidth)
            ms = data.get('markersize', markersize)
            zorder = data.get('zorder', i)

            if is_x_string:
                # Check if the data contains time strings in format 'HH:MM'
                if all(':' in str(x) for x in x_raw):
                    # Convert time strings to minutes for plotting
                    def time_to_minutes(time_str):
                        hours, minutes = map(int, time_str.split(':'))
                        return hours * 60 + minutes

                    x_plot = np.array([time_to_minutes(x) for x in x_raw])
                    # Update global time limits
                    current_min_x, current_max_x = np.nanmin(
                        x_plot), np.nanmax(x_plot)
                    if not np.isnan(current_min_x):
                        min_x_global = min(min_x_global, current_min_x)
                    if not np.isnan(current_max_x):
                        max_x_global = max(max_x_global, current_max_x)
                else:
                    # Fallback to index-based plotting for non-time strings
                    x_plot = np.arange(len(x_raw))
            else:
                x_plot = x_raw
                # Update global numeric X limits
                current_min_x, current_max_x = np.nanmin(x_raw), np.nanmax(
                    x_raw)
                if not np.isnan(current_min_x):
                    min_x_global = min(min_x_global, current_min_x)
                if not np.isnan(current_max_x):
                    max_x_global = max(max_x_global, current_max_x)
            epist, alea = data.get('epist', None), data.get('alea', None)
            if epist is not None:
                line = ax.fill_between(x_plot,
                                       y - 1.96 * epist,
                                       y + 1.96 * epist,
                                       color='#9bed87',
                                       alpha=0.3,
                                       zorder=zorder - 2,
                                       label='Epistemic uncertainty')
                if 'Epistemic uncertainty' not in unique_labels:
                    unique_labels['Epistemic uncertainty'] = line
            if alea is not None:
                line = ax.fill_between(
                    x_plot,
                    y - 1.96 * alea,
                    y + 1.96 * alea,
                    color='#879bed',
                    alpha=0.3,
                    zorder=zorder - 3,
                    label='数据不确定性' if epist is not None else '不确定性')
                if 'Aleatoric uncertainty' not in unique_labels:
                    unique_labels['Aleatoric uncertainty'] = line
            if alea is not None and epist is not None:
                total_sigma = np.sqrt((epist**2) + (alea**2))
                line = ax.fill_between(x_plot,
                                       y - 1.96 * total_sigma,
                                       y + 1.96 * total_sigma,
                                       color='#edd987',
                                       alpha=0.3,
                                       zorder=zorder - 4,
                                       label='Total uncertainty')
                if 'Total uncertainty' not in unique_labels:
                    unique_labels['Total uncertainty'] = line
            line, = ax.plot(x_plot,
                            y,
                            label=label,
                            color=color,
                            linestyle=linestyle,
                            marker=marker,
                            linewidth=lw,
                            markersize=ms if marker != 'None' else 0,
                            zorder=zorder)

            if label not in unique_labels: unique_labels[label] = line

            # current_min_y, current_max_y = np.nanmin(y), np.nanmax(y)
            # if not np.isnan(current_min_y):
            #     min_y_global = min(min_y_global, current_min_y)
            # if not np.isnan(current_max_y):
            #     max_y_global = max(max_y_global, current_max_y)

    # --- Apply Global Settings with Padding ---
    shared_ax = axes[0, 0]
    # Set Y limits with padding
    if np.isfinite(min_y_global) and np.isfinite(max_y_global):
        y_range = max_y_global - min_y_global
        y_pad = y_range * axis_limit_padding
        shared_ax.set_ylim(min_y_global - y_pad, max_y_global + y_pad)
    else:
        shared_ax.set_ylim(0, 1)

    # Set X limits with padding
    if is_x_string:
        # Check if it's time format
        if all(':' in str(x) for x in x_values_list_global):
            # For time strings, limits are set in the tick formatting section
            pass  # Limits will be set there
        else:
            # Non-time strings, use index-based limits
            shared_ax.set_xlim(min_x_idx_global - 5, max_x_idx_global + 0.5)
    elif np.isfinite(min_x_global) and np.isfinite(max_x_global):
        x_range = max_x_global - min_x_global
        x_pad = x_range * axis_limit_padding if x_range > 1e-9 else 0.1
        shared_ax.set_xlim(min_x_global - x_pad, max_x_global + x_pad)
    else:
        if max_x_idx_global > min_x_idx_global:
            shared_ax.set_xlim(min_x_idx_global - 0.5, max_x_idx_global + 0.5)
        else:
            shared_ax.set_xlim(0, 1)

    # --- Apply MaxNLocator and Format Ticks ---
    # Only apply MaxNLocator for non-time string data
    if not (is_x_string and all(':' in str(x) for x in x_values_list_global)):
        shared_ax.xaxis.set_major_locator(plt.MaxNLocator(num_xticks))

    # Format X ticks if x data was string-based time format
    if is_x_string and x_values_list_global:
        try:
            # Check if the data contains time strings in format 'HH:MM'
            if all(':' in str(x) for x in x_values_list_global):
                # Set fixed x-axis limits for time format (00:00 to 25:59)
                # Convert time strings to minutes for easier calculation
                def time_to_minutes(time_str):
                    hours, minutes = map(int, time_str.split(':'))
                    return hours * 60 + minutes

                def minutes_to_time(minutes):
                    hours = minutes // 60
                    mins = minutes % 60
                    return f'{hours:02d}:{mins:02d}'

                # Set fixed limits: 00:00 to max time from data
                min_minutes = 0  # 00:00
                max_minutes = max_x_global  # Use actual max from data

                # Set x-axis limits with 1 hour padding on left to avoid overlap with y-axis
                # and no padding on right so max tick aligns with right edge
                shared_ax.set_xlim(min_minutes - 60,
                                   max_minutes)  # 1 hour padding only on left

                # Set fixed tick positions at 00:00, 09:00, and 18:00
                # Only show ticks that are within the data range
                tick_positions = []
                tick_labels = []

                # Always include 00:00
                tick_positions.append(0)
                tick_labels.append('00:00')

                # Add 09:00 if it's within range
                if 9 * 60 <= max_minutes:
                    tick_positions.append(9 * 60)
                    tick_labels.append('09:00')

                # Add 18:00 if it's within range
                if 18 * 60 <= max_minutes:
                    tick_positions.append(18 * 60)
                    tick_labels.append('18:00')

                # Don't add max time tick - let the right edge be clean

                shared_ax.set_xticks(tick_positions)
                shared_ax.set_xticklabels(tick_labels, rotation=0)
            else:
                # Fallback to original method for non-time string data
                tick_indices = shared_ax.get_xticks()
                valid_indices = [
                    int(round(idx)) for idx in tick_indices
                    if min_x_idx_global <= idx <= max_x_idx_global + 1e-6
                ]  # Round index
                tick_labels = [
                    x_values_list_global[idx] for idx in valid_indices
                    if 0 <= idx < len(x_values_list_global)
                ]
                shared_ax.set_xticks(valid_indices)
                shared_ax.set_xticklabels(tick_labels,
                                          rotation=0)  # Ensure no rotation
        except Exception as e:
            print(
                f"Warning: Could not format string x-ticks: {e}. Using index labels."
            )

    # --- Apply Styling and Manually Control Labels/Tick Labels ---
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            ax.tick_params(axis='both',
                           which='both',
                           labelsize=tick_fontsize,
                           direction='in',
                           pad=tick_pad,
                           length=tick_length,
                           top=True,
                           right=True)
            if use_grid:
                ax.grid(True,
                        which='both',
                        linestyle='--',
                        color='gray',
                        linewidth=0.5)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

            # --- MODIFICATION: Clear individual subplot labels ---
            ax.set_ylabel('')
            ax.set_xlabel('')

            # Y-axis Tick Labels: Show only for the first column
            if c != 0: ax.tick_params(axis='y', labelleft=False)
            # X-axis Tick Labels: Show only for the last row
            if r != nrows - 1: ax.tick_params(axis='x', labelbottom=False)

    # --- MODIFICATION: Add Figure-level Labels ---
    fig.supylabel(default_ylabel, fontsize=label_fontsize,
                  x=label_pad)  # Adjust x for left padding
    fig.supxlabel(default_xlabel, fontsize=label_fontsize,
                  y=label_pad)  # Adjust y for bottom padding

    # --- Shared Legend ---
    handles = list(unique_labels.values())
    labels = list(unique_labels.keys())
    bottom_margin_dynamic = 0.05  # Default bottom margin if no legend

    if handles:
        final_legend_ncol = legend_ncol
        if final_legend_ncol is None:
            max_per_row = math.floor(figure_width /
                                     (legend_fontsize / 10 * 1.5))
            final_legend_ncol = min(len(handles), max(1, int(max_per_row)))

        legend_rows = math.ceil(len(labels) / final_legend_ncol)
        # Use tight multiplier (1.3) and base margin (0.03) for bottom margin calculation
        if total_height > 0:
            bottom_margin_dynamic = 0.03 + (legend_fontsize / 72 * 1.3 *
                                            legend_rows) / total_height
        else:
            bottom_margin_dynamic = 0.1 + 0.05 * legend_rows

        fig.legend(
            handles,
            labels,
            loc=legend_loc,  # 'upper center'
            bbox_to_anchor=
            legend_bbox_to_anchor,  # Use updated default (e.g., (0.5, -0.01))
            ncol=final_legend_ncol,
            fontsize=legend_fontsize,
            frameon=False,
            borderaxespad=0.)

    # --- Final Layout Adjustment (Compact Spacing) ---
    # Adjust top margin slightly to give titles space, and use tight hspace
    plt.subplots_adjust(
        left=0.1,
        right=0.95,
        bottom=bottom_margin_dynamic,
        top=0.93,  # Adjust top slightly for titles
        wspace=0.05,
        hspace=subplot_hspace)  # Compact spacing

    # --- Save or Display ---
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        full_save_path = f"{save_path}.{file_format}"
        # Use bbox_inches='tight' when saving to ensure legend is included properly
        plt.savefig(full_save_path,
                    dpi=dpi,
                    bbox_inches='tight',
                    format=file_format)
        print(f"-> 🔍 Figure saved to {full_save_path}")
        plt.close(fig)
    else:
        plt.show()

    return fig, axes


def calculate_r2(group, col_1, col_2):
    y_true = group[col_1]
    y_pred = group[col_2]
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    ss_res = np.sum((y_true - y_pred)**2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan


def plot_roadtype_r2(
    preds_df,
    save_path,
    horizon,
    highways={
        1: 'Motorway',
        2: 'Motorway link',
        3: 'Trunk',
        4: 'Trunk link',
        5: 'Primary',
        6: 'Primary link',
        7: 'Secondary',
        8: 'Secondary link',
        9: 'Tertiary',
        10: 'Tertiary link',
        11: 'Busway',
        12: 'Residential',
        13: 'Living street',
        14: 'Unclassified'
    }):
    """
    Plot the node R² coverage curve for different road types, with shared axes and Arial font, and save as a PDF.

    Args:
        preds_df: DataFrame containing 'node', 'roadtype', 'true', 'absolute error' columns.
        save_path: The path to save the figure (e.g., 'roadtype_r2.pdf').
        highways: Optional dict mapping roadtype to display name.
    """
    plt.rcParams['font.family'] = 'Arial'
    # Calculate R² for each (node, roadtype) group
    roadtype_r2 = preds_df.groupby(['node',
                                    'roadtype']).apply(calculate_r2,
                                                       col_1=f'True_{horizon}',
                                                       col_2=f'Pred_{horizon}')
    roadtype_r2 = roadtype_r2.loc[roadtype_r2 > 0]
    roadtypes = roadtype_r2.reset_index().roadtype.unique()
    n_types = len(roadtypes)
    n_cols = 3
    n_rows = (n_types + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows,
                             n_cols,
                             figsize=(6, 6 / 1.7),
                             sharex=True,
                             sharey=True,
                             dpi=200)
    axes = axes.flatten()

    for idx, roadtype in enumerate(roadtypes):
        ax = axes[idx]
        # Extract and sort R² values for current roadtype
        type_r2 = roadtype_r2.xs(roadtype, level='roadtype')
        sorted_r2 = np.sort(type_r2)[::-1]
        y = np.arange(1, len(sorted_r2) + 1) / len(sorted_r2) * 100

        ax.plot(sorted_r2, y, 'b-', linewidth=2)

        # Annotate key coverage points
        key_points = [25, 50, 75, 90]
        for point in key_points:
            idx_point = int(len(sorted_r2) * point / 100)
            if idx_point < len(sorted_r2):
                ax.plot(sorted_r2[idx_point], point, 'ro')
                ax.annotate(f'{sorted_r2[idx_point]:.3f}',
                            xy=(sorted_r2[idx_point], point),
                            xytext=(10, 0),
                            textcoords='offset points',
                            ha='left',
                            va='center',
                            fontsize=9,
                            fontname='Arial')

        # Use mapped name if highways is provided
        title = f'{highways.get(roadtype, roadtype)} (Num: {len(type_r2)})' if highways else f'{roadtype} (Num: {len(type_r2)})'
        ax.set_title(title, fontsize=10, fontname='Arial')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.label_outer()  # Only show outer tick labels

    # Remove unused subplots
    for idx in range(len(roadtypes), len(axes)):
        fig.delaxes(axes[idx])

    # Set shared axis labels for the whole figure
    fig.text(0.5,
             0.04,
             'R²',
             ha='center',
             va='center',
             fontsize=12,
             fontname='Arial')
    fig.text(0.06,
             0.5,
             'Good Node Coverage (%)',
             ha='center',
             va='center',
             rotation='vertical',
             fontsize=12,
             fontname='Arial')

    plt.tight_layout(rect=[0.08, 0.06, 1, 0.98])
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"-> 🔍 Figure saved to {save_path}")


def plot_node_r2(preds_df, save_path, horizon) -> set:
    """
    Plot the cumulative distribution of R² across all nodes and save as a PDF file.

    Args:
        preds_df: DataFrame containing 'node', 'true', 'absolute error' columns.
        save_path: Path to save the figure (e.g., 'node_r2_curve.pdf').
    """
    # Set global font to Arial and sizes for SCI standard
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10

    # Calculate R² for all nodes

    node_r2 = preds_df.groupby('node').apply(calculate_r2,
                                             col_1=f'True_{horizon}',
                                             col_2=f'Pred_{horizon}')
    bad_nodes = set(list(node_r2.loc[node_r2 < 0].index))
    node_r2 = node_r2.loc[node_r2 > 0]
    # Create figure
    fig = plt.figure(figsize=(6, 6 / 1.7), dpi=200)

    # Sort R²
    sorted_r2 = np.sort(node_r2)[::-1]
    y = np.arange(1, len(sorted_r2) + 1) / len(sorted_r2) * 100

    # Plot curve
    plt.plot(sorted_r2, y, 'b-', linewidth=2)

    # Annotate key points
    key_points = [25, 50, 75, 90]  # coverage percentiles
    for point in key_points:
        idx_point = int(len(sorted_r2) * point / 100)
        if idx_point < len(sorted_r2):
            plt.plot(sorted_r2[idx_point], point, 'ro')
            plt.annotate(f'{sorted_r2[idx_point]:.3f}',
                         xy=(sorted_r2[idx_point], point),
                         xytext=(10, 0),
                         textcoords='offset points',
                         ha='left',
                         va='center',
                         fontsize=10,
                         fontname='Arial')

    # Set axis labels and title
    plt.xlabel('R²', fontsize=10, fontname='Arial')
    plt.ylabel('Good Node Coverage (%)', fontsize=10, fontname='Arial')
    plt.title(
        f'Cumulative Distribution of R² across All Good Nodes (Num: {len(node_r2)})',
        fontsize=12,
        fontname='Arial')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"-> 🔍 Figure saved to {save_path}")
    return bad_nodes


def plot_node_type_bar(node_groups, save_path):
    """
    Draw a bar chart comparing the counts of two node types (e.g., good nodes and bad nodes)
    using SCI paper standards (Arial font, suitable sizes, high DPI).

    Args:
        node_groups: A list with two tuples, each is (node_type_name, node_id_list)
                     e.g. [('Good nodes', [1,2,3]), ('Bad nodes', [5,6])]
        save_path: Path to save the figure (e.g., 'node_type_bar.pdf')
    """
    # SCI journal style
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10

    labels = [item[0] for item in node_groups]
    counts = [len(item[1]) for item in node_groups]

    fig, ax = plt.subplots(figsize=(4, 4 / 1.7), dpi=200)

    bars = ax.bar(labels, counts, color=['#4575b4', '#d73027'], width=0.6)

    # Annotate bar values
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f'{int(height)}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=10,
            fontname='Arial')

    ax.set_ylabel('Count', fontsize=12, fontname='Arial')
    ax.set_xlabel('Node Type', fontsize=12, fontname='Arial')
    ax.set_title('Node Count by Type', fontsize=12, fontname='Arial')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"-> 🔍 Figure saved to {save_path}")


def calibration_plot(p_values, save_path):

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

    # histogram
    plt.figure(figsize=(6, 6 / 1.7), dpi=300)
    plt.hist(p_values,
             bins=20,
             density=True,
             edgecolor='black',
             label='Observed p-value distribution')

    plt.axhline(y=1,
                color='red',
                linestyle='--',
                linewidth=2,
                label='Uniform distribution')

    plt.xlabel('Predicted cumulative probability',
               fontsize=12,
               fontname='Arial')
    plt.ylabel('Density', fontsize=12, fontname='Arial')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 1)
    plt.savefig(save_path + '_histogram.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> 🔍 Figure saved to {save_path}_histogram.pdf")

    # p-p curve
    plt.figure(figsize=(6, 6 / 1.7), dpi=300)
    x_axis = np.linspace(0, 1, 101)
    y_axis = []
    for p_theoretical in x_axis:
        proportion = np.mean(p_values <= p_theoretical)
        y_axis.append(proportion)
    plt.plot(x_axis, y_axis, 'b-', label='Model calibration curve')
    plt.xlabel('Predicted cumulative probability',
               fontsize=12,
               fontname='Arial')
    plt.ylabel('Empirical cumulative probability',
               fontsize=12,
               fontname='Arial')
    plt.plot([0, 1], [0, 1],
             'gray',
             linewidth=2,
             label='Perfect calibration line',
             linestyle='-.')
    plt.legend()
    plt.savefig(save_path + '_curve.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> 🔍 Figure saved to {save_path}_curve.pdf")


def node_calibration_distribution(node_qice_scores, node_cv_scores, to_obs_min,
                                  relative_node_cv_scores,
                                  avg_mode_neighbor_diff, save_path):
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

    plt.figure(figsize=(6, 6 / 1.7), dpi=300)
    plt.hist(node_qice_scores, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('QICE score', fontsize=12, fontname='Arial')
    plt.ylabel('Number of nodes', fontsize=12, fontname='Arial')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path + '_qice.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    if to_obs_min is not None:
        plt.figure(figsize=(6, 6 / 1.7), dpi=300)
        plt.scatter(to_obs_min, node_qice_scores, alpha=0.5, s=3)
        plt.xlabel('Dist. to nearest observed node',
                   fontsize=12,
                   fontname='Arial')
        plt.ylabel('QICE score', fontsize=12, fontname='Arial')
        plt.tight_layout()
        plt.savefig(save_path + '_qice_to_d2o.pdf',
                    dpi=300,
                    bbox_inches='tight')
        plt.close()
        print(f"-> 🔍 Figure saved to {save_path}_qice_to_d2o.pdf")

    plt.figure(figsize=(6, 6 / 1.7), dpi=300)
    plt.scatter(avg_mode_neighbor_diff, node_qice_scores, alpha=0.5, s=3)
    plt.xlabel('Neighbor homophily', fontsize=12, fontname='Arial')
    plt.ylabel('Calibration error (QICE)', fontsize=12, fontname='Arial')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path + '_neighbor_homophily.pdf',
                dpi=300,
                bbox_inches='tight')
    plt.close()
    print(f"-> 🔍 Figure saved to {save_path}_neighbor_homophily.pdf")

    plt.figure(figsize=(6, 6 / 1.7), dpi=300)
    plt.scatter(relative_node_cv_scores, node_qice_scores, alpha=0.5, s=3)
    plt.xlabel('Relative Confidence', fontsize=12, fontname='Arial')
    plt.ylabel('Calibration error (QICE)', fontsize=12, fontname='Arial')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path + '_relative_confidence.pdf',
                dpi=300,
                bbox_inches='tight')
    plt.close()
    print(f"-> 🔍 Figure saved to {save_path}_relative_confidence.pdf")

    plt.figure(figsize=(6, 6 / 1.7), dpi=300)
    plt.hist(node_cv_scores, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Coefficient of Variation', fontsize=12, fontname='Arial')
    plt.ylabel('Number of nodes', fontsize=12, fontname='Arial')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path + '_cv.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> 🔍 Figure saved to {save_path}_cv.pdf")


def caculate_qice(p_values_1d, M=10):
    """
    Calculate the QICE score
    Args:
        p_values_1d (np.array): 1D array containing all p-values
        M (int): Number of bins.

    Returns:
        float: The QICE score
    """
    if len(p_values_1d) == 0:
        return np.nan

    counts, _ = np.histogram(p_values_1d, bins=M, range=(0, 1))

    proportions = counts / len(p_values_1d)

    ideal_proportion = 1.0 / M

    qice_score = np.mean(np.abs(proportions - ideal_proportion))

    return qice_score


def node_qice_distribution_subplot(qice_scores, ax, title):
    """Plot QICE distribution on a subplot"""
    ax.hist(qice_scores, bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('QICE score', fontsize=10, fontname='Arial')
    ax.set_ylabel('Number of nodes', fontsize=10, fontname='Arial')
    ax.set_title(title, fontsize=12, fontname='Arial')
    ax.grid(True, linestyle='--', alpha=0.7)


def cv_distribution_subplot(cv_scores, ax, title):
    """Plot CV distribution on a subplot"""
    ax.hist(cv_scores, bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Coefficient of Variation', fontsize=10, fontname='Arial')
    ax.set_ylabel('Number of nodes', fontsize=10, fontname='Arial')
    ax.set_title(title, fontsize=12, fontname='Arial')
    ax.grid(True, linestyle='--', alpha=0.7)


def calibration_plot_subplot(p_values, ax, title):
    """Plot calibration histogram on a subplot"""
    ax.hist(p_values,
            bins=20,
            density=True,
            edgecolor='black',
            alpha=0.7,
            label='Observed p-value distribution')
    ax.axhline(y=1,
               color='red',
               linestyle='--',
               linewidth=2,
               label='Uniform distribution')
    ax.set_xlabel('Predicted cumulative probability',
                  fontsize=10,
                  fontname='Arial')
    ax.set_ylabel('Density', fontsize=10, fontname='Arial')
    ax.set_title(title, fontsize=12, fontname='Arial')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)


def calibration_curve_subplot(p_values, ax, title):
    """Plot calibration curve on a subplot"""
    x_axis = np.linspace(0, 1, 101)
    y_axis = []
    for p_theoretical in x_axis:
        proportion = np.mean(p_values <= p_theoretical)
        y_axis.append(proportion)
    ax.plot(x_axis, y_axis, 'b-', label='Model calibration curve')
    ax.set_xlabel('Predicted cumulative probability',
                  fontsize=10,
                  fontname='Arial')
    ax.set_ylabel('Empirical cumulative probability',
                  fontsize=10,
                  fontname='Arial')
    ax.plot([0, 1], [0, 1],
            'gray',
            linewidth=2,
            label='Perfect calibration line',
            linestyle='-.')
    ax.set_title(title, fontsize=12, fontname='Arial')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_road_uncertainty_analysis(
    sub_df,
    horizon_absolute,
    output_dir,
    has_epist=False,
    highways={
        1: 'Motorway',
        2: 'Motorway link',
        3: 'Trunk',
        4: 'Trunk link',
        5: 'Primary',
        6: 'Primary link',
        7: 'Secondary',
        8: 'Secondary link',
        9: 'Tertiary',
        10: 'Tertiary link',
        11: 'Busway',
        12: 'Residential',
        13: 'Living street',
        14: 'Unclassified'
    }):
    """
    Plot uncertainty analysis for all road types in a single figure with subplots.

    Args:
        sub_df: DataFrame containing the data
        horizon: Horizon value for analysis
        output_dir: Directory to save the figures
        has_epist: Whether epistemic uncertainty is available
    """
    # Get unique roads
    roads = sub_df.roadtype.unique()
    n_roads = len(roads)

    # Set up matplotlib parameters
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

    # Calculate optimal subplot layout for A4 paper
    # A4 width: 210mm = 8.27 inches, considering margins: ~6.5 inches usable width
    # A4 height: 297mm = 11.69 inches, considering margins: ~9 inches usable height
    max_cols = 3  # Maximum 3 columns to fit A4 width
    n_cols = min(n_roads, max_cols)
    n_rows = (n_roads + n_cols - 1) // n_cols  # Ceiling division

    # Calculate figure size to fit A4 paper
    fig_width = min(6.5, n_cols *
                    2.0)  # Max 6.5 inches width, each subplot ~2 inches
    fig_height = min(9.0, n_rows *
                     2.5)  # Max 9 inches height, each subplot ~2.5 inches

    # Create three separate figures with subplots for each road

    # Figure 1: QICE distribution for Alea
    fig_qice_alea, axes_qice_alea = plt.subplots(n_rows,
                                                 n_cols,
                                                 figsize=(fig_width,
                                                          fig_height),
                                                 dpi=300)
    if n_roads == 1:
        axes_qice_alea = np.array([axes_qice_alea])
    elif n_rows == 1:
        axes_qice_alea = axes_qice_alea.reshape(1, -1)
    else:
        axes_qice_alea = axes_qice_alea.reshape(n_rows, n_cols)

    # Figure 2: Calibration histogram for Alea
    fig_cal_hist_alea, axes_cal_hist_alea = plt.subplots(n_rows,
                                                         n_cols,
                                                         figsize=(fig_width,
                                                                  fig_height),
                                                         dpi=300)
    if n_roads == 1:
        axes_cal_hist_alea = np.array([axes_cal_hist_alea])
    elif n_rows == 1:
        axes_cal_hist_alea = axes_cal_hist_alea.reshape(1, -1)
    else:
        axes_cal_hist_alea = axes_cal_hist_alea.reshape(n_rows, n_cols)

    # Figure 3: Calibration curve for Alea
    fig_cal_curve_alea, axes_cal_curve_alea = plt.subplots(
        n_rows, n_cols, figsize=(fig_width, fig_height), dpi=300)
    if n_roads == 1:
        axes_cal_curve_alea = np.array([axes_cal_curve_alea])
    elif n_rows == 1:
        axes_cal_curve_alea = axes_cal_curve_alea.reshape(1, -1)
    else:
        axes_cal_curve_alea = axes_cal_curve_alea.reshape(n_rows, n_cols)

    # Figure 4: CV distribution for Alea
    fig_cv_alea, axes_cv_alea = plt.subplots(n_rows,
                                             n_cols,
                                             figsize=(fig_width, fig_height),
                                             dpi=300)
    if n_roads == 1:
        axes_cv_alea = np.array([axes_cv_alea])
    elif n_rows == 1:
        axes_cv_alea = axes_cv_alea.reshape(1, -1)
    else:
        axes_cv_alea = axes_cv_alea.reshape(n_rows, n_cols)

    # Initialize figures for epistemic uncertainty if needed
    fig_qice_epist = None
    fig_cal_hist_epist = None
    fig_cal_curve_epist = None
    fig_cv_epist = None
    fig_qice_total = None
    fig_cal_hist_total = None
    fig_cal_curve_total = None
    fig_cv_total = None
    axes_qice_epist = None
    axes_cal_hist_epist = None
    axes_cal_curve_epist = None
    axes_cv_epist = None
    axes_qice_total = None
    axes_cal_hist_total = None
    axes_cal_curve_total = None
    axes_cv_total = None

    if has_epist:
        fig_qice_epist, axes_qice_epist = plt.subplots(n_rows,
                                                       n_cols,
                                                       figsize=(fig_width,
                                                                fig_height),
                                                       dpi=300)
        if n_roads == 1:
            axes_qice_epist = np.array([axes_qice_epist])
        elif n_rows == 1:
            axes_qice_epist = axes_qice_epist.reshape(1, -1)
        else:
            axes_qice_epist = axes_qice_epist.reshape(n_rows, n_cols)

        fig_cal_hist_epist, axes_cal_hist_epist = plt.subplots(
            n_rows, n_cols, figsize=(fig_width, fig_height), dpi=300)
        if n_roads == 1:
            axes_cal_hist_epist = np.array([axes_cal_hist_epist])
        elif n_rows == 1:
            axes_cal_hist_epist = axes_cal_hist_epist.reshape(1, -1)
        else:
            axes_cal_hist_epist = axes_cal_hist_epist.reshape(n_rows, n_cols)

        fig_cal_curve_epist, axes_cal_curve_epist = plt.subplots(
            n_rows, n_cols, figsize=(fig_width, fig_height), dpi=300)
        if n_roads == 1:
            axes_cal_curve_epist = np.array([axes_cal_curve_epist])
        elif n_rows == 1:
            axes_cal_curve_epist = axes_cal_curve_epist.reshape(1, -1)
        else:
            axes_cal_curve_epist = axes_cal_curve_epist.reshape(n_rows, n_cols)

        fig_cv_epist, axes_cv_epist = plt.subplots(n_rows,
                                                   n_cols,
                                                   figsize=(fig_width,
                                                            fig_height),
                                                   dpi=300)
        if n_roads == 1:
            axes_cv_epist = np.array([axes_cv_epist])
        elif n_rows == 1:
            axes_cv_epist = axes_cv_epist.reshape(1, -1)
        else:
            axes_cv_epist = axes_cv_epist.reshape(n_rows, n_cols)

        fig_qice_total, axes_qice_total = plt.subplots(n_rows,
                                                       n_cols,
                                                       figsize=(fig_width,
                                                                fig_height),
                                                       dpi=300)
        if n_roads == 1:
            axes_qice_total = np.array([axes_qice_total])
        elif n_rows == 1:
            axes_qice_total = axes_qice_total.reshape(1, -1)
        else:
            axes_qice_total = axes_qice_total.reshape(n_rows, n_cols)

        fig_cal_hist_total, axes_cal_hist_total = plt.subplots(
            n_rows, n_cols, figsize=(fig_width, fig_height), dpi=300)
        if n_roads == 1:
            axes_cal_hist_total = np.array([axes_cal_hist_total])
        elif n_rows == 1:
            axes_cal_hist_total = axes_cal_hist_total.reshape(1, -1)
        else:
            axes_cal_hist_total = axes_cal_hist_total.reshape(n_rows, n_cols)

        fig_cal_curve_total, axes_cal_curve_total = plt.subplots(
            n_rows, n_cols, figsize=(fig_width, fig_height), dpi=300)
        if n_roads == 1:
            axes_cal_curve_total = np.array([axes_cal_curve_total])
        elif n_rows == 1:
            axes_cal_curve_total = axes_cal_curve_total.reshape(1, -1)
        else:
            axes_cal_curve_total = axes_cal_curve_total.reshape(n_rows, n_cols)
        fig_cv_total, axes_cv_total = plt.subplots(n_rows,
                                                   n_cols,
                                                   figsize=(fig_width,
                                                            fig_height),
                                                   dpi=300)
        if n_roads == 1:
            axes_cv_total = np.array([axes_cv_total])
        elif n_rows == 1:
            axes_cv_total = axes_cv_total.reshape(1, -1)
        else:
            axes_cv_total = axes_cv_total.reshape(n_rows, n_cols)

        # Plot for each road
    for i, road in enumerate(roads):
        road_df = sub_df.loc[sub_df.roadtype == road]

        # Calculate row and column indices for the subplot
        row_idx = i // n_cols
        col_idx = i % n_cols

        # Alea uncertainty plots
        alea_p_values = road_df[f'AleaP_{horizon_absolute}']
        node_qice_scores = road_df.groupby(
            'node')[f'AleaP_{horizon_absolute}'].apply(caculate_qice)
        node_cv_scores = road_df.groupby(
            'node')[f'AleaCV_{horizon_absolute}'].mean()

        node_qice_distribution_subplot(node_qice_scores,
                                       axes_qice_alea[row_idx, col_idx],
                                       f'{highways.get(road, road)}')
        calibration_plot_subplot(alea_p_values, axes_cal_hist_alea[row_idx,
                                                                   col_idx],
                                 f'{highways.get(road, road)}')
        calibration_curve_subplot(alea_p_values, axes_cal_curve_alea[row_idx,
                                                                     col_idx],
                                  f'{highways.get(road, road)}')
        cv_distribution_subplot(node_cv_scores, axes_cv_alea[row_idx, col_idx],
                                f'{highways.get(road, road)}')

        # Epistemic uncertainty plots if available
        if has_epist:
            epist_p_values = road_df[f'EpistP_{horizon_absolute}']
            node_qice_scores = road_df.groupby(
                'node')[f'EpistP_{horizon_absolute}'].apply(caculate_qice)
            node_cv_scores = road_df.groupby(
                'node')[f'EpistCV_{horizon_absolute}'].mean()
            node_qice_distribution_subplot(node_qice_scores,
                                           axes_qice_epist[row_idx, col_idx],
                                           f'{highways.get(road, road)}')
            calibration_plot_subplot(epist_p_values,
                                     axes_cal_hist_epist[row_idx, col_idx],
                                     f'{highways.get(road, road)}')
            calibration_curve_subplot(epist_p_values,
                                      axes_cal_curve_epist[row_idx, col_idx],
                                      f'{highways.get(road, road)}')
            cv_distribution_subplot(node_cv_scores, axes_cv_epist[row_idx,
                                                                  col_idx],
                                    f'{highways.get(road, road)}')
            total_uncertainty_p_values = road_df[f'TUP_{horizon_absolute}']
            node_qice_scores = road_df.groupby(
                'node')[f'TUP_{horizon_absolute}'].apply(caculate_qice)
            node_cv_scores = road_df.groupby(
                'node')[f'TUCV_{horizon_absolute}'].mean()
            node_qice_distribution_subplot(node_qice_scores,
                                           axes_qice_total[row_idx, col_idx],
                                           f'{highways.get(road, road)}')
            calibration_plot_subplot(total_uncertainty_p_values,
                                     axes_cal_hist_total[row_idx, col_idx],
                                     f'{highways.get(road, road)}')
            calibration_curve_subplot(total_uncertainty_p_values,
                                      axes_cal_curve_total[row_idx, col_idx],
                                      f'{highways.get(road, road)}')
            cv_distribution_subplot(node_cv_scores, axes_cv_total[row_idx,
                                                                  col_idx],
                                    f'{highways.get(road, road)}')

    # Hide empty subplots
    for i in range(n_roads, n_rows * n_cols):
        row_idx = i // n_cols
        col_idx = i % n_cols
        axes_qice_alea[row_idx, col_idx].set_visible(False)
        axes_cal_hist_alea[row_idx, col_idx].set_visible(False)
        axes_cal_curve_alea[row_idx, col_idx].set_visible(False)
        axes_cv_alea[row_idx, col_idx].set_visible(False)
        if has_epist:
            axes_qice_epist[row_idx, col_idx].set_visible(False)
            axes_cal_hist_epist[row_idx, col_idx].set_visible(False)
            axes_cal_curve_epist[row_idx, col_idx].set_visible(False)
            axes_cv_epist[row_idx, col_idx].set_visible(False)
            axes_qice_total[row_idx, col_idx].set_visible(False)
            axes_cal_hist_total[row_idx, col_idx].set_visible(False)
            axes_cal_curve_total[row_idx, col_idx].set_visible(False)
            axes_cv_total[row_idx, col_idx].set_visible(False)

    # Save the figures
    fig_qice_alea.tight_layout()
    fig_qice_alea.savefig(output_dir +
                          f"_node_qice_distribution_alea_all_roads.pdf",
                          dpi=300,
                          bbox_inches='tight')
    plt.close(fig_qice_alea)
    # Print the path where the Alea node QICE distribution figure is saved
    print(
        f"-> 🔍 Figure saved to {output_dir}_node_qice_distribution_alea_all_roads.pdf"
    )

    fig_cal_hist_alea.tight_layout()
    fig_cal_hist_alea.savefig(output_dir +
                              f"_calibration_histogram_alea_all_roads.pdf",
                              dpi=300,
                              bbox_inches='tight')
    plt.close(fig_cal_hist_alea)
    print(
        f"-> 🔍 Figure saved to {output_dir}_calibration_histogram_alea_all_roads.pdf"
    )
    fig_cal_curve_alea.tight_layout()
    fig_cal_curve_alea.savefig(output_dir +
                               f"_calibration_curve_alea_all_roads.pdf",
                               dpi=300,
                               bbox_inches='tight')
    plt.close(fig_cal_curve_alea)
    print(
        f"-> 🔍 Figure saved to {output_dir}_calibration_curve_alea_all_roads.pdf"
    )
    fig_cv_alea.tight_layout()
    fig_cv_alea.savefig(output_dir + f"_cv_distribution_alea_all_roads.pdf",
                        dpi=300,
                        bbox_inches='tight')
    plt.close(fig_cv_alea)
    print(
        f"-> 🔍 Figure saved to {output_dir}_cv_distribution_alea_all_roads.pdf"
    )
    if has_epist:
        fig_qice_epist.tight_layout()
        fig_qice_epist.savefig(output_dir +
                               f"_node_qice_distribution_epist_all_roads.pdf",
                               dpi=300,
                               bbox_inches='tight')
        plt.close(fig_qice_epist)
        print(
            f"-> 🔍 Figure saved to {output_dir}_node_qice_distribution_epist_all_roads.pdf"
        )
        fig_cal_hist_epist.tight_layout()
        fig_cal_hist_epist.savefig(
            output_dir + f"_calibration_histogram_epist_all_roads.pdf",
            dpi=300,
            bbox_inches='tight')
        plt.close(fig_cal_hist_epist)
        print(
            f"-> 🔍 Figure saved to {output_dir}_calibration_histogram_epist_all_roads.pdf"
        )
        fig_cal_curve_epist.tight_layout()
        fig_cal_curve_epist.savefig(output_dir +
                                    f"_calibration_curve_epist_all_roads.pdf",
                                    dpi=300,
                                    bbox_inches='tight')
        plt.close(fig_cal_curve_epist)
        print(
            f"-> 🔍 Figure saved to {output_dir}_calibration_curve_epist_all_roads.pdf"
        )

        fig_cv_epist.tight_layout()
        fig_cv_epist.savefig(output_dir +
                             f"_cv_distribution_epist_all_roads.pdf",
                             dpi=300,
                             bbox_inches='tight')
        plt.close(fig_cv_epist)
        print(
            f"-> 🔍 Figure saved to {output_dir}_cv_distribution_epist_all_roads.pdf"
        )
        fig_qice_total.tight_layout()
        fig_qice_total.savefig(output_dir +
                               f"_node_qice_distribution_total_all_roads.pdf",
                               dpi=300,
                               bbox_inches='tight')
        plt.close(fig_qice_total)
        print(
            f"-> 🔍 Figure saved to {output_dir}_node_qice_distribution_total_all_roads.pdf"
        )
        fig_cal_hist_total.tight_layout()
        fig_cal_hist_total.savefig(
            output_dir + f"_calibration_histogram_total_all_roads.pdf",
            dpi=300,
            bbox_inches='tight')
        plt.close(fig_cal_hist_total)
        print(
            f"-> 🔍 Figure saved to {output_dir}_calibration_histogram_total_all_roads.pdf"
        )
        fig_cal_curve_total.tight_layout()
        fig_cal_curve_total.savefig(output_dir +
                                    f"_calibration_curve_total_all_roads.pdf",
                                    dpi=300,
                                    bbox_inches='tight')
        plt.close(fig_cal_curve_total)
        print(
            f"-> 🔍 Figure saved to {output_dir}_calibration_curve_total_all_roads.pdf"
        )
        fig_cv_total.tight_layout()
        fig_cv_total.savefig(output_dir +
                             f"_cv_distribution_total_all_roads.pdf",
                             dpi=300,
                             bbox_inches='tight')
        plt.close(fig_cv_total)
        print(
            f"-> 🔍 Figure saved to {output_dir}_cv_distribution_total_all_roads.pdf"
        )


def plot_cv_vs_error(cv_values, ape_values, save_path):
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

    plot_df = pd.DataFrame({
        'uncertainty_cv': cv_values,
        'error_ape': ape_values
    })
    try:
        plot_df['uncertainty_bin'] = pd.qcut(plot_df['uncertainty_cv'],
                                             q=10,
                                             labels=False,
                                             duplicates='drop')
    except Exception as e:
        print(f"Warning: qcut failed ({e}), falling back to cut.")
        plot_df['uncertainty_bin'] = pd.cut(plot_df['uncertainty_cv'],
                                            bins=10,
                                            labels=False)

    grouped = plot_df.groupby('uncertainty_bin')['uncertainty_cv']

    bin_ranges = grouped.agg(['min', 'max'])

    bin_labels = [
        f"{row['min']:.2f}-{row['max']:.2f}"
        for index, row in bin_ranges.iterrows()
    ]
    plt.figure(figsize=(6, 6 / 1.7), dpi=300)
    sns.boxplot(x='uncertainty_bin',
                y='error_ape',
                data=plot_df,
                showfliers=False)

    # Compute statistics for each bin and connect the points.
    bin_stats = plot_df.groupby('uncertainty_bin')['error_ape'].agg(
        ['median', lambda x: x.quantile(0.25),
         lambda x: x.quantile(0.75)]).reset_index()
    bin_stats.columns = ['uncertainty_bin', 'median', 'q1', 'q3']

    # Plot the three lines.
    x_positions = range(len(bin_labels))
    plt.plot(x_positions,
             bin_stats['median'],
             'r-',
             linewidth=2,
             label='Median',
             marker='o',
             markersize=6)
    plt.plot(x_positions,
             bin_stats['q1'],
             'b--',
             linewidth=2,
             label='Q1 (25%)',
             marker='s',
             markersize=6)
    plt.plot(x_positions,
             bin_stats['q3'],
             'g--',
             linewidth=2,
             label='Q3 (75%)',
             marker='^',
             markersize=6)

    plt.xticks(range(len(bin_labels)),
               bin_labels,
               rotation=45,
               fontsize=10,
               fontname='Arial')
    plt.xlabel('Predicted coefficient of variation bins',
               fontsize=12,
               fontname='Arial')
    plt.ylabel('Distribution of absolute percentage error',
               fontsize=12,
               fontname='Arial')
    plt.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> 🔍 Figure saved to {save_path}")


def plot_z_distribution(z_values, save_path):
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.figure(figsize=(6, 6 / 1.7), dpi=300)
    plt.hist(z_values, bins=100, edgecolor='black', alpha=0.7, density=True)
    plt.xlabel('Z', fontsize=12, fontname='Arial')
    plt.ylabel('Proportion of nodes', fontsize=12, fontname='Arial')
    plt.grid(True, linestyle='--', alpha=0.7)
    x = np.linspace(-7, 7, 100)
    y = norm.pdf(x, loc=0, scale=1)
    plt.plot(x, y, 'r-', label='Normal Distribution', linewidth=2)
    plt.tight_layout()
    plt.savefig(save_path + '_z_distribution.pdf',
                dpi=300,
                bbox_inches='tight')
    plt.close()
    print(f"-> 🔍 Figure saved to {save_path}_z_distribution.pdf")
