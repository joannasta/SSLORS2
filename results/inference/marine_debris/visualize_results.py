import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import json
import os
cwd = os.getcwd()
print(cwd)
path = cwd + '/results/inference/marine_debris/'
data_marida = json.load(open(path+"results_marida.json"))
data = {"marida": data_marida }

# Grouped Bar Chart (Agia Napa)
models = list(data["marida"].keys())
metrics = list(data["marida"][ "Frozen_embedding"].keys())
values = [[data["marida"][model][metric] for model in models] for metric in metrics]

x = range(len(models))
width = 0.2

plt.figure(figsize=(10, 6))
for i, metric in enumerate(metrics):
    plt.bar([pos + i * width for pos in x], values[i], width, label=metric)

plt.xticks([pos + width for pos in x], models, rotation=45, ha='right')
plt.legend()
plt.title("Model Performance Comparison")
plt.ylabel("Metric Value")
plt.tight_layout()
plt.savefig("Marida_BarChart.png")
plt.close()

# Grouped Bar Chart (Agia Napa)
models = list(data["marida"].keys())
metrics = list(data["marida"][ "Frozen_embedding"].keys())
values = [[data["marida"][model][metric] for model in models] for metric in metrics]

x = range(len(models))
width = 0.2

plt.figure(figsize=(10, 6))
for i, metric in enumerate(metrics):
    plt.bar([pos + i * width for pos in x], values[i], width, label=metric)

plt.xticks([pos + width for pos in x], models, rotation=45, ha='right')
plt.legend()
plt.title("Model Performance Comparison ")
plt.ylabel("Metric Value")
plt.tight_layout()
plt.savefig("Marida_BarChart.png")
plt.close()

# Line Charts (Metric Trends)
metrics = list(data["marida"]["Frozen_embedding"].keys())
models = list(data["marida"].keys())
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



# Delta Scatter Plot of MAE and RMSE
plt.figure(figsize=(8, 6))
for region in regions:
    my_iou = data[region][ "Frozen_embedding"]["IoU"]
    paper_iou = data[region][ "Unet_marida"]["IoU"]
    my_pa = data[region][ "Frozen_embedding"]["PA"]
    paper_pa = data[region][ "Unet_marida"]["PA"]
    my_f1 = data[region][ "Frozen_embedding"]["PA"]
    paper_f1 = data[region][ "Unet_marida"]["PA"]
    plt.scatter(my_iou, paper_pa, my_f1,label=f"{region} IoU")
    plt.scatter(my_iou, paper_pa, my_f1,label=f"{region} PA")

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