import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_model_comparison():
    df = pd.read_csv('results/summary_table.csv')
    df = df.sort_values('f1', ascending=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    metrics = [
        ('loss', 'Loss (Lower is Better)', 'Reds_r'),
        ('accuracy', 'Accuracy', 'Blues'),
        ('precision', 'Precision', 'Greens'),
        ('recall', 'Recall', 'Purples')
    ]
    
    for ax, (metric, title, cmap) in zip(axes.flatten(), metrics):
        if metric in df.columns and df[metric].notna().any():
            colors = plt.cm.get_cmap(cmap)(np.linspace(0.4, 0.8, len(df)))
            bars = ax.bar(df['model'], df[metric], color=colors, alpha=0.8, edgecolor='black')
            
            ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
            ax.set_xlabel('Model', fontsize=11, fontweight='bold')
            ax.set_ylabel(title.split()[0], fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # En iyi modeli vurgula
            if metric != 'loss':
                best_idx = df[metric].idxmax()
            else:
                best_idx = df[metric].idxmin()
            bars[list(df.index).index(best_idx)].set_edgecolor('gold')
            bars[list(df.index).index(best_idx)].set_linewidth(3)
            
            # Değerleri yaz
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    print("Grafik kaydedildi: results/model_comparison.png")
    plt.close()


def plot_f1_ranking():
    df = pd.read_csv('results/summary_table.csv')
    df = df.sort_values('f1', ascending=True)  # Horizontal bar için artan sıralama
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Renk gradyanı
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df)))
    bars = ax.barh(df['model'], df['f1'], color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Model Ranking by F1 Score', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim([0, 1.0])
    
    # En iyi modeli vurgula
    bars[-1].set_edgecolor('gold')
    bars[-1].set_linewidth(3)
    
    # Değerleri yaz
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
               f' {width:.4f}',
               ha='left', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/f1_ranking.png', dpi=300, bbox_inches='tight')
    print("Grafik kaydedildi: results/f1_ranking.png")
    plt.close()


def plot_precision_recall_f1():
    df = pd.read_csv('results/summary_table.csv')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    scatter = ax.scatter(df['precision'], df['recall'], 
                        s=df['f1']*500, alpha=0.6, 
                        c=range(len(df)), cmap='viridis',
                        edgecolors='black', linewidth=2)
    
    # Model isimlerini ekle
    for _, row in df.iterrows():
        ax.annotate(row['model'], 
                   (row['precision'], row['recall']),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    
    ax.set_xlabel('Precision', fontsize=12, fontweight='bold')
    ax.set_ylabel('Recall', fontsize=12, fontweight='bold')
    ax.set_title('Precision vs Recall (Bubble size = F1 Score)', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0.5, 1.0])
    ax.set_ylim([0.5, 1.0])
    
    # Diagonal line (perfect balance)
    ax.plot([0.5, 1.0], [0.5, 1.0], 'r--', alpha=0.3, linewidth=2, label='Perfect Balance')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/precision_recall_scatter.png', dpi=300, bbox_inches='tight')
    print("Grafik kaydedildi: results/precision_recall_scatter.png")
    plt.close()


def plot_radar_chart():
    df = pd.read_csv('results/summary_table.csv')
    
    # Metrikleri normalize et (0-1 arası)
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Her model için
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the plot
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for idx, row in df.iterrows():
        values = [row[m] for m in metrics]
        values += values[:1]  # Close the plot
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['model'], 
               color=colors[idx % len(colors)], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.capitalize() for m in metrics], fontsize=11, fontweight='bold')
    ax.set_ylim(0.5, 1.0)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.grid(True, linestyle='--', alpha=0.4)
    
    ax.set_title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/radar_chart.png', dpi=300, bbox_inches='tight')
    print("Grafik kaydedildi: results/radar_chart.png")
    plt.close()


def plot_metric_heatmap():
    df = pd.read_csv('results/summary_table.csv')
    
    # Metrikleri seç
    metrics_df = df.set_index('model')[['accuracy', 'precision', 'recall', 'f1']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(metrics_df.T, annot=True, fmt='.4f', cmap='YlGnBu', 
                cbar_kws={'label': 'Score'}, ax=ax,
                linewidths=2, linecolor='white', vmin=0.5, vmax=1.0)
    
    ax.set_title('Model Performance Heatmap', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/metric_heatmap.png', dpi=300, bbox_inches='tight')
    print("Grafik kaydedildi: results/metric_heatmap.png")
    plt.close()


def visualize_all():
    """Tüm görselleştirmeleri oluştur"""
    
    print("\n" + "="*80)
    print("SONUÇLAR GÖRSELLEŞTİRİLİYOR")
    print("="*80 + "\n")
    
    plot_model_comparison()
    plot_f1_ranking()
    plot_precision_recall_f1()
    plot_radar_chart()
    plot_metric_heatmap()
    
    print("\n" + "="*80)
    print("TÜM GÖRSELLEŞTİRMELER TAMAMLANDI!")
    print("="*80)
    print("\nOluşturulan dosyalar:")
    print("  results/model_comparison.png")
    print("  results/f1_ranking.png")
    print("  results/precision_recall_scatter.png")
    print("  results/radar_chart.png")
    print("  results/metric_heatmap.png")


if __name__ == "__main__":
    visualize_all()