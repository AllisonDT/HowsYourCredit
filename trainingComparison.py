import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_names = ["mlp/training_mlp.csv", "cnn/training_cnn.csv", "transformers/training_transformers.csv", "lstm/training_lstm.csv"]
model_names = ["MLP", "CNN", "Transformers", "LSTM"]

selected_metrics = ['val_accuracy', 'precision', 'recall', 'f1_score']

metrics_list = []
for file in file_names:
    df = pd.read_csv(file)
    metrics = df[selected_metrics].iloc[0].tolist()
    metrics_list.append(metrics)
    
metrics_data = np.array(metrics_list)


# Bar Chart
def plot_grouped_bar_chart(metric_labels, metrics_data, model_names):
    num_metrics = len(metric_labels)
    num_models = len(model_names)
    x = np.arange(num_metrics)
    width = 0.2 
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i in range(num_models):
        offset = (i - num_models/2) * width + width/2
        ax.bar(x + offset, metrics_data[i], width, label=model_names[i])
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Comparison of ML Models: Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()

    plt.savefig("training_bar_chart.png", dpi=300, bbox_inches='tight')
    plt.show()

# Radar Chart
def plot_radar_chart(metric_labels, metrics_data, model_names):
    num_vars = len(metric_labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for model, data in zip(model_names, metrics_data):
        values = data.tolist()
        values += values[:1]
        ax.plot(angles, values, label=model)
        ax.fill(angles, values, alpha=0.1)

    ax.set_title('Radar Chart of ML Models Performance Metrics', y=1.1, fontsize=15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.savefig("training_radar_chart.png", dpi=300, bbox_inches='tight')
    plt.show()

plot_grouped_bar_chart(selected_metrics, metrics_data, model_names)
plot_radar_chart(selected_metrics, metrics_data, model_names)
