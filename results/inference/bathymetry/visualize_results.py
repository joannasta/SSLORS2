import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

data = {
    "agia_napa": {
        "baseline": {"avg_test_mae": 0.503, "avg_test_rmse": 0.6632, "avg_test_std_dev": 0.6139},
        "random": {"avg_test_mae": 0.7107, "avg_test_rmse": 0.9194, "avg_test_std_dev": 0.7052},
        "fullyfinetuned": {"avg_test_mae": 0.5578, "avg_test_rmse": 0.7251, "avg_test_std_dev": 0.6602},
        "paper_results": {"avg_test_mae": 1.068, "avg_test_rmse": 0.694, "avg_test_std_dev": 0.94},
    },
    "puck_lagoon": {
        "baseline": {"avg_test_mae": 0.744, "avg_test_rmse": 1.0448, "avg_test_std_dev": 0.7963},
        "random": {"avg_test_mae": 0.6464, "avg_test_rmse": 0.9587, "avg_test_std_dev": 0.835},
        "fullyfinetuned": {"avg_test_mae": 0.4425, "avg_test_rmse": 0.8295, "avg_test_std_dev": 0.7184},
        "paper_results": {"avg_test_mae": 0.907, "avg_test_rmse": 0.493, "avg_test_std_dev": 0.874},
    },
}

# Grouped Bar Chart (Agia Napa)
models = list(data["agia_napa"].keys())
metrics = list(data["agia_napa"]["baseline"].keys())
values = [[data["agia_napa"][model][metric] for model in models] for metric in metrics]

x = range(len(models))
width = 0.2

plt.figure(figsize=(10, 6))
for i, metric in enumerate(metrics):
    plt.bar([pos + i * width for pos in x], values[i], width, label=metric)

plt.xticks([pos + width for pos in x], models, rotation=45, ha='right')
plt.legend()
plt.title("Model Performance Comparison - Agia Napa")
plt.ylabel("Metric Value")
plt.tight_layout()
plt.savefig("Agia_Napa_BarChart.png")
plt.close()

# Grouped Bar Chart (Agia Napa)
models = list(data["puck_lagoon"].keys())
metrics = list(data["puck_lagoon"]["baseline"].keys())
values = [[data["puck_lagoon"][model][metric] for model in models] for metric in metrics]

x = range(len(models))
width = 0.2

plt.figure(figsize=(10, 6))
for i, metric in enumerate(metrics):
    plt.bar([pos + i * width for pos in x], values[i], width, label=metric)

plt.xticks([pos + width for pos in x], models, rotation=45, ha='right')
plt.legend()
plt.title("Model Performance Comparison - Puck Lagoon")
plt.ylabel("Metric Value")
plt.tight_layout()
plt.savefig("Puck_Lagoon_BarChart.png")
plt.close()

# Line Charts (Metric Trends)
metrics = list(data["agia_napa"]["baseline"].keys())
models = list(data["agia_napa"].keys())
regions = list(data.keys())

plt.figure(figsize=(12, 8))
for i, metric in enumerate(metrics):
    plt.subplot(2, 2, i + 1)
    for region in regions:
        values = [data[region][model][metric] for model in models]
        plt.plot(models, values, label=region)
    plt.title(f"{metric.upper()} Trends Across Models")
    plt.xlabel("Models")
    plt.ylabel(metric.upper())
    plt.xticks(rotation=45, ha='right')
    plt.legend()
plt.tight_layout()
plt.savefig("Metric_Trends_LineCharts.png")
plt.close()

# Heatmap Tables
for region in regions:
    region_data = data[region]
    df = pd.DataFrame.from_dict(region_data, orient='index')
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, cmap="YlGnBu")
    plt.title(f"Model Performance Heatmap - {region.capitalize()}")
    plt.tight_layout()
    plt.savefig(f"{region.capitalize()}_Heatmap.png")
    plt.close()


# Delta Bar Chart of MAE
delta_mae = {}

for region in regions:
    my_mae = data[region]["baseline"]["avg_test_mae"]
    paper_mae = data[region]["paper_results"]["avg_test_mae"]
    delta_mae[region] = my_mae - paper_mae

plt.figure(figsize=(8, 6))
plt.bar(delta_mae.keys(), delta_mae.values())
plt.title("Delta MAE: My Baseline vs. Paper Results")
plt.ylabel("Delta MAE")
plt.tight_layout()
plt.savefig("Delta_MAE_BarChart.png")
plt.close()

# Delta Scatter Plot of MAE and RMSE
plt.figure(figsize=(8, 6))
for region in regions:
    my_mae = data[region]["baseline"]["avg_test_mae"]
    paper_mae = data[region]["paper_results"]["avg_test_mae"]
    my_rmse = data[region]["baseline"]["avg_test_rmse"]
    paper_rmse = data[region]["paper_results"]["avg_test_rmse"]
    plt.scatter(my_mae, paper_mae, label=f"{region} MAE")
    plt.scatter(my_rmse, paper_rmse, label=f"{region} RMSE")

plt.plot([min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])],
         [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])],
         color='r', linestyle='--')
plt.xlabel("My Results")
plt.ylabel("Paper Results")
plt.title("Comparison: My Baseline vs. Paper Results")
plt.legend()
plt.tight_layout()
plt.savefig("Results_Scatter_Comparison.png")
plt.close()