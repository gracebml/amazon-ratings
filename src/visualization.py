import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List


def plot_rating_distribution(ratings: np.ndarray, title: str = "Phân Bố Rating") -> None:
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Bar chart
    unique_ratings, counts = np.unique(ratings, return_counts=True)
    percentages = counts / len(ratings) * 100
    
    colors = ['#d62728', '#ff7f0e', '#ffbb78', '#98df8a', '#2ca02c']
    axes[0].bar(unique_ratings, counts, color=colors, edgecolor='black', alpha=0.8)
    axes[0].set_xlabel('Rating', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Số lượng', fontsize=13, fontweight='bold')
    axes[0].set_title(title, fontsize=15, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add labels
    for r, c, p in zip(unique_ratings, counts, percentages):
        axes[0].text(r, c, f'{p:.1f}%\n{c:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Pie chart sentiment
    positive = (ratings >= 4).sum()
    neutral = (ratings == 3).sum()
    negative = (ratings <= 2).sum()
    
    sentiment_labels = ['Positive\n(4-5★)', 'Neutral\n(3★)', 'Negative\n(1-2★)']
    sentiment_counts = [positive, neutral, negative]
    sentiment_colors = ['#2ca02c', '#ffbb78', '#d62728']
    
    axes[1].pie(sentiment_counts, labels=sentiment_labels, colors=sentiment_colors,
               autopct='%1.1f%%', startangle=90, explode=(0.05, 0, 0.05),
               textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[1].set_title('Phân Loại Sentiment', fontsize=15, fontweight='bold')
    
    # 3. Histogram with mean/median
    axes[2].hist(ratings, bins=20, color='steelblue', edgecolor='black', alpha=0.7, density=True)
    axes[2].axvline(np.mean(ratings), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(ratings):.2f}')
    axes[2].axvline(np.median(ratings), color='green', linestyle='--', linewidth=2, label=f'Median = {np.median(ratings):.2f}')
    axes[2].set_xlabel('Rating', fontsize=13, fontweight='bold')
    axes[2].set_ylabel('Density', fontsize=13, fontweight='bold')
    axes[2].set_title('Distribution', fontsize=15, fontweight='bold')
    axes[2].legend(fontsize=11)
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_user_analysis(user_counts: np.ndarray, 
                      user_group_masks: dict,
                      ratings: np.ndarray,
                      user_inverse: np.ndarray) -> None:
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Histogram số ratings per user
    axes[0, 0].hist(user_counts, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(user_counts), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(user_counts):.1f}')
    axes[0, 0].axvline(np.median(user_counts), color='green', linestyle='--', linewidth=2, label=f'Median = {np.median(user_counts):.0f}')
    axes[0, 0].set_xlabel('Số ratings per user', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Số lượng users (log scale)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Distribution: Ratings per User', fontsize=14, fontweight='bold')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Pie chart phân loại users
    user_group_counts = [np.sum(mask) for mask in user_group_masks.values()]
    user_group_labels = list(user_group_masks.keys())
    colors_users = ['#d62728', '#ff7f0e', '#ffbb78', '#98df8a']
    
    axes[0, 1].pie(user_group_counts, labels=user_group_labels, colors=colors_users,
                  autopct='%1.1f%%', startangle=90,
                  textprops={'fontsize': 11, 'fontweight': 'bold'})
    axes[0, 1].set_title('Phân Loại Users', fontsize=14, fontweight='bold')
    
    # 3. Boxplot rating by user group
    group_data = []
    group_names = []
    for group_name, group_mask in user_group_masks.items():
        group_user_indices = np.where(group_mask)[0]
        group_rating_mask = np.isin(user_inverse, group_user_indices)
        group_ratings = ratings[group_rating_mask]
        if len(group_ratings) > 0:
            group_data.append(group_ratings)
            group_names.append(group_name)
    
    bp = axes[1, 0].boxplot(group_data, labels=group_names, patch_artist=True,
                            boxprops=dict(facecolor='lightblue', alpha=0.7),
                            medianprops=dict(color='red', linewidth=2))
    axes[1, 0].set_ylabel('Rating', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Rating Distribution by User Group', fontsize=14, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 4. CDF
    sorted_counts = np.sort(user_counts)
    cdf = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    axes[1, 1].plot(sorted_counts, cdf, linewidth=2, color='darkblue')
    axes[1, 1].axhline(0.8, color='red', linestyle='--', alpha=0.7, label='80% users')
    percentile_80 = sorted_counts[int(0.8*len(sorted_counts))]
    axes[1, 1].axvline(percentile_80, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].set_xlabel('Số ratings per user', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('CDF: Ratings per User', fontsize=14, fontweight='bold')
    axes[1, 1].set_xscale('log')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_temporal_analysis(data_temporal: np.ndarray, ratings: np.ndarray) -> None:
    
    years = data_temporal['year']
    months = data_temporal['month']
    day_of_week = data_temporal['day_of_week']
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Time series
    ax1 = fig.add_subplot(gs[0, :])
    year_month = years * 100 + months
    unique_ym, ym_inverse = np.unique(year_month, return_inverse=True)
    ym_counts = np.bincount(ym_inverse)
    ym_rating_sums = np.bincount(ym_inverse, weights=ratings)
    ym_avg_ratings = ym_rating_sums / ym_counts
    
    ym_years = unique_ym // 100
    ym_months = unique_ym % 100
    ym_labels = [f"{y}-{m:02d}" for y, m in zip(ym_years, ym_months)]
    
    x = np.arange(len(unique_ym))
    ax1_twin = ax1.twinx()
    
    ax1.bar(x, ym_counts, alpha=0.3, color='steelblue', label='Số ratings')
    ax1.set_ylabel('Số lượng ratings', fontsize=12, fontweight='bold', color='steelblue')
    
    ax1_twin.plot(x, ym_avg_ratings, color='red', linewidth=2, marker='o', markersize=4, label='Avg rating')
    ax1_twin.set_ylabel('Average rating', fontsize=12, fontweight='bold', color='red')
    ax1_twin.set_ylim([1, 5])
    
    ax1.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax1.set_title('Trend: Ratings Over Time', fontsize=15, fontweight='bold')
    ax1.set_xticks(x[::3])
    ax1.set_xticklabels([ym_labels[i] for i in range(0, len(ym_labels), 3)], rotation=45, ha='right')
    ax1.grid(alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)
    ax1_twin.legend(loc='upper right', fontsize=10)
    
    # 2. By month
    ax2 = fig.add_subplot(gs[1, 0])
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_idx = months - 1
    month_counts = np.bincount(month_idx, minlength=12)
    month_rating_sums = np.bincount(month_idx, weights=ratings, minlength=12)
    month_avg_ratings = np.zeros(12)
    valid_mask = month_counts > 0
    month_avg_ratings[valid_mask] = month_rating_sums[valid_mask] / month_counts[valid_mask]
    
    bars = ax2.bar(month_names, month_avg_ratings, color='coral', alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Average rating', fontsize=12, fontweight='bold')
    ax2.set_title('Average Rating by Month', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 5])
    ax2.axhline(np.mean(ratings), color='blue', linestyle='--', linewidth=2, label='Overall mean')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. By day of week
    ax3 = fig.add_subplot(gs[1, 1])
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_counts = np.bincount(day_of_week, minlength=7)
    dow_rating_sums = np.bincount(day_of_week, weights=ratings, minlength=7)
    dow_avg_ratings = np.zeros(7)
    dow_valid = dow_counts > 0
    dow_avg_ratings[dow_valid] = dow_rating_sums[dow_valid] / dow_counts[dow_valid]
    
    colors_dow = ['steelblue']*5 + ['coral', 'coral']
    ax3.bar(dow_names, dow_avg_ratings, color=colors_dow, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Average rating', fontsize=12, fontweight='bold')
    ax3.set_title('Average Rating by Day of Week', fontsize=14, fontweight='bold')
    ax3.set_ylim([0, 5])
    ax3.axhline(np.mean(ratings), color='blue', linestyle='--', linewidth=2, label='Overall mean')
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Heatmap
    ax4 = fig.add_subplot(gs[2, :])
    heatmap_data = np.zeros((12, 7))
    for m in range(1, 13):
        for d in range(7):
            mask = (months == m) & (day_of_week == d)
            if np.sum(mask) > 0:
                heatmap_data[m-1, d] = np.mean(ratings[mask])
    
    im = ax4.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=1, vmax=5)
    ax4.set_xticks(np.arange(7))
    ax4.set_yticks(np.arange(12))
    ax4.set_xticklabels(dow_names)
    ax4.set_yticklabels(month_names)
    ax4.set_xlabel('Day of Week', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Month', fontsize=12, fontweight='bold')
    ax4.set_title('Heatmap: Average Rating (Month × Day of Week)', fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax4, label='Avg Rating')
    plt.tight_layout()
    plt.show()


def plot_comparison_normalization(data: np.ndarray, 
                                  data_normalized: np.ndarray,
                                  data_standardized: np.ndarray,
                                  data_log: np.ndarray) -> None:
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Original
    axes[0, 0].hist(data['Rating'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Original Rating', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Rating')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].text(0.05, 0.95, f'μ={np.mean(data["Rating"]):.2f}\nσ={np.std(data["Rating"]):.2f}',
                    transform=axes[0, 0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Min-Max
    axes[0, 1].hist(data_normalized['Rating_normalized'], bins=20, color='coral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Min-Max Normalized', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Rating (normalized)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].text(0.05, 0.95, f'μ={np.mean(data_normalized["Rating_normalized"]):.2f}\nσ={np.std(data_normalized["Rating_normalized"]):.2f}',
                    transform=axes[0, 1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Z-score
    axes[1, 0].hist(data_standardized['Rating_normalized'], bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('Z-score Standardized', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Rating (standardized)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].text(0.05, 0.95, f'μ={np.mean(data_standardized["Rating_normalized"]):.2f}\nσ={np.std(data_standardized["Rating_normalized"]):.2f}',
                    transform=axes[1, 0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Log
    axes[1, 1].hist(data_log['Rating_normalized'], bins=20, color='plum', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Log Transformed', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Rating (log)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].text(0.05, 0.95, f'μ={np.mean(data_log["Rating_normalized"]):.2f}\nσ={np.std(data_log["Rating_normalized"]):.2f}',
                    transform=axes[1, 1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

