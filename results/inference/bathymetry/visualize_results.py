import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import json
import os
cwd = os.getcwd()
print(cwd)
path = cwd + '/results/inference/bathymetry/'
data_agia_napa = json.load(open(path+"agia_napa_10000.json"))
data_puck_lagoon = json.load(open(path+"puck_lagoon_10000.json"))
data = {"agia_napa": data_agia_napa, "puck_lagoon": data_puck_lagoon}

# Grouped Bar Chart (Agia Napa)
models = list(data["agia_napa"].keys())
metrics = list(data["agia_napa"][  "Frozen_embedding"].keys())
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
metrics = list(data["puck_lagoon"]["Frozen_embedding"].keys())
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
metrics = list(data["agia_napa"]["Frozen_embedding"].keys())
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
    my_mae = data[region]["Frozen_embedding"]["avg_test_mae"]
    paper_mae = data[region][  "Unet_magicbathynet"]["avg_test_mae"]
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
    my_mae = data[region]["Frozen_embedding"]["avg_test_mae"]
    paper_mae = data[region][  "Unet_magicbathynet"]["avg_test_mae"]
    my_rmse = data[region]["Frozen_embedding"]["avg_test_rmse"]
    paper_rmse = data[region][  "Unet_magicbathynet"]["avg_test_rmse"]
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