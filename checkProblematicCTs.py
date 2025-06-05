#%%
import nibabel as nib
import os
import numpy as np
import pandas as pd
#%%
resultsDf = pd.read_csv("/home/p/ppetrd/mukerjeset2/results200.csv")
firstOrderDf = pd.read_csv("/home/p/ppetrd/mukerjeset2/firstOrderResults200.csv")
#take last 18 columns of resultsDf with column names
firstOrderColumnsToConcat = firstOrderDf[firstOrderDf.columns[-18:]]
finalDf = pd.concat([resultsDf, firstOrderColumnsToConcat], axis=1)
finalDf.to_csv("/home/p/ppetrd/mukerjeset2/finalResults200.csv", index=False)
# %%
finalDfSet1 = pd.read_csv("/home/p/ppetrd/mukerjeset1/finalResults100.csv")
finalDfSet2 = pd.read_csv("/home/p/ppetrd/mukerjeset2/finalResults100.csv")
all_results = pd.concat([finalDfSet1, finalDfSet2], axis=0, ignore_index=True)
all_results.to_csv("/home/p/ppetrd/mukerjeset2/finalResultsAll100.csv", index=False)
checkdf = pd.read_csv("/home/p/ppetrd/mukerjeset2/finalResultsAll100.csv")

# %%
df_2 = pd.read_csv("/home/p/ppetrd/mukerjeset2/finalResultsAll2.csv", index_col=0).dropna(axis=1, how='all')
df_5 = pd.read_csv("/home/p/ppetrd/mukerjeset2/finalResultsAll5.csv", index_col=0).dropna(axis=1, how='all')
df_25 = pd.read_csv("/home/p/ppetrd/mukerjeset2/finalResultsAll.csv", index_col=0).dropna(axis=1, how='all')
df_50 = pd.read_csv("/home/p/ppetrd/mukerjeset2/finalResultsAll50.csv", index_col=0).dropna(axis=1, how='all')
df_100 = pd.read_csv("/home/p/ppetrd/mukerjeset2/finalResultsAll100.csv", index_col=0).dropna(axis=1, how='all')
df_200 = pd.read_csv("/home/p/ppetrd/mukerjeset2/finalResultsAll200.csv", index_col=0).dropna(axis=1, how='all')
dfs = {
    2: df_2,  # DataFrame for binWidth=2
    5: df_5,  # DataFrame for binWidth=5
    25: df_25,  # DataFrame for binWidth=25
    50: df_50,  # DataFrame for binWidth=50
    100: df_100,
    200: df_200
}

# Stack feature values for each patient across bin widths
features = df_25.columns[38:]  # Assuming features start from column index 38
patients = df_25.index

# Store average CV for each feature
feature_stability = {}

for feature in features:
    values = np.stack([dfs[bw][feature].values for bw in dfs], axis=1)
    # Skip if all values are NaN
    if np.all(np.isnan(values)):
        continue
    # Calculate CV only for non-zero means
    means = np.mean(values, axis=1)
    stds = np.std(values, axis=1)
    # Only calculate CV where mean is not near zero
    valid_mask = np.abs(means) > 1e-10
    if np.any(valid_mask):
        cv_per_patient = np.full(len(means), np.nan)
        cv_per_patient[valid_mask] = stds[valid_mask] / np.abs(means[valid_mask])
        avg_cv = np.nanmean(cv_per_patient)
        if not np.isnan(avg_cv):
            feature_stability[feature] = avg_cv
        else:
            print(f"Warning: avg  '{feature}' are NaN for this patient.")
    else:
        feature_stability[feature] = np.nan

# Lower avg_cv means higher stability (less variability across bin widths)
print("Feature stability (average CV):")
for feature, avg_cv in feature_stability.items():
    print(f"{feature}: {avg_cv:.3f}")


# %%
