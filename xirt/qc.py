"""Module to perform QC on the xiRT performance."""
import pandas as pd

#todo#
# epoch learning plot

#todo
# eval plot tain val pred
summary_df = collect_summaries(dir, grid_dir=False)
order = (summary_df[summary_df["fold"] == "Pred"]).groupby(["fold", "setup"]).agg(np.mean)[
    "loss"].sort_values().reset_index()["setup"]
f, ax = plt.subplots(1)
ax = sns.barplot(x="fold", y="loss", hue="setup", data=summary_df,
                 order=["Train", "Val", "Pred"], hue_order=order, ax=ax)
sns.despine()
plt.show()
save_fig(f, outpath + "Fig3d_barplot")

# fold plots
df_epoch_history = pd.read_excel("results/qc_data/epoch_history.xls")
df_predictions = pd.read_excel("results/qc_data/prediction.xls")
df_errors = pd.read_excel("results/qc_data/errors.xls")
df_errors = pd.read_excel("results/qc_data/errors.xls")



def figure3b():
    param = {"output": {"scx-metrics": "acc", "hsax-metrics": "acc"}}
    summary_df = pd.read_pickle(summary_loc)
    n_folds = summary_df.fold.value_counts().Train
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5, 4))
    ax1 = sns.stripplot(x="fold", y="SCX_" + param["output"]["scx-metrics"], data=summary_df,
                        ax=ax1, color=SCX_color)
    ax1 = sns.pointplot(x="fold", y="SCX_" + param["output"]["scx-metrics"], data=summary_df,
                        join=True, color="grey", ax=ax1, scale=0.7)
    ax1.set(title="SCX", ylabel="Accuracy", ylim=(0, 1), xlabel="")
    sns.despine()
    # next
    ax2 = sns.stripplot(x="fold", y="hSAX_" + param["output"]["hsax-metrics"], data=summary_df,
                        ax=ax2, color=hSAX_color)
    ax2 = sns.pointplot(x="fold", y="hSAX_" + param["output"]["hsax-metrics"], data=summary_df,
                        join=True, color="grey", ax=ax2, scale=0.7)

    ax2.set(title="hSAX", ylabel="Accuracy",
            xlabel="Fold type\n(k-fold: {})".format(n_folds), ylim=(0, 1))
    ax3 = sns.stripplot(x="fold", y="RP_mse", data=summary_df, ax=ax3, color=RP_color)
    ax3 = sns.pointplot(x="fold", y="RP_mse", data=summary_df, join=True, color="grey", ax=ax3,
                        scale=0.7)
    ax3.set(title="RP", ylabel="mean squared error (MSE)", xlabel="")

    for ax in (ax1, ax2, ax3):
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    sns.despine()
    f.tight_layout()
    save_fig(f, outpath + "b_stripplot")