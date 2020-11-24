"""Module to perform QC on the xiRT performance."""
import glob
import logging
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import palettable
import pandas as pd
import seaborn as sns
import statannot
from matplotlib import ticker
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report

sns.set(context="notebook", style="white", palette="deep", font_scale=1)

# constants
hSAX_color = "#9b59b6"
SCX_color = "#3498db"
RP_color = "#e74c3c"
colors = [hSAX_color, SCX_color, RP_color, "C1", "C2", "C7"]
colormaps = [plt.cm.Purples, plt.cm.Blues, plt.cm.Reds,
             plt.cm.Oranges, plt.cm.Greens, plt.cm.Greys]
targetdecoys_cm = sns.xkcd_palette(["faded green", "orange", "dark orange"])
TOTAL_color = palettable.cartocolors.qualitative.Bold_6.mpl_colors[1:3][1]
new_rc_params = {'text.usetex': False, "svg.fonttype": 'none'}
mpl.rcParams.update(new_rc_params)
logger = logging.getLogger(__name__)


def encode_pval(pvalue):
    """
    Given a p-value returns a significant annotation of the float in * decoding.

    According to: * <= 0.05, ** <= 0.01, *** <= 0.001, **** <= 0.0001

    Args:
        pvalue: float, p-value

    Returns:
        str, annotation
    """
    reference_pval = np.array([0.0001, 0.001, 0.01, 0.05, 1])
    reference_str = ["****", "****", "***", "**", "*", "ns"]
    return reference_str[np.where(reference_pval > pvalue)[0][0]]


def statistical_annotation(x1, x2, yvalues, txt, ax):  # pragma: no cover
    """
    Add annotation to axes plot.

    Args:
        x1:
        x2:
        yvalues:
        txt: str, text to add
        ax: axes object, matplotlib axes

    Returns:
        None
    """
    # statistical annotation
    y, h, col = yvalues.max() + 2, 2, 'k'
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    ax.text((x1 + x2) * .5, y + h, txt, ha='center', va='bottom', color=col)


def add_heatmap(y, yhat, task, ax, colormap, dims):  # pragma: no cover
    """Generate a heatmap to visualize classification results and return plot to given axes.

    Args:
        y: ar-like, observed
        yhat: ar-like, predicted
        task: ar-like, task names
        ax: axes, matplotlib axes
        colormap: colormap, plt.cm colormap instance
        dims: ar-like, unique dimensions for the input data

    Returns:
        axes, matplotlib axes object
    """
    # prepare confusion matrix
    # note that yhat and y are change in the order here! This is needed because sklearn
    # arranges the confusion matrix very confusing ... this way bottom left, to top right
    # will equal a correaltion / straight line = expected behavior
    cm_scx = pd.DataFrame(np.flip(confusion_matrix(yhat, y), axis=0))
    cm_scx.columns = cm_scx.columns
    cm_scx.index = np.flip(cm_scx.columns)
    # mask for not writing zeroes into the plot
    mask = cm_scx <= 0
    cm_scx[mask] = np.nan
    # annotation
    metric_str = """r2: {:.2f} f1: {:.2f} acc: {:.2f} racc: {:.2f}""".format(
        custom_r2(y, yhat), f1_score(y, yhat, average="macro"),
        accuracy_score(y, yhat), relaxed_accuracy(y, yhat))
    metric_str = """r2: {:.2f} f1: {:.2f} acc: {:.2f} racc: {:.2f}""".format(
        custom_r2(y, yhat), f1_score(y, yhat, average="macro"),
        accuracy_score(y, yhat), relaxed_accuracy(y, yhat))
    logger.info("QC: {}".format(task))
    logger.info("Metrics: {}".format(metric_str))
    ax = sns.heatmap(cm_scx, cmap=colormap, annot=True, annot_kws={"size": 12},
                     fmt='.0f', cbar=True, mask=mask, ax=ax)
    ax.axhline(y=dims[-1], color='k')
    ax.axvline(x=0, color='k')
    ax.set(ylim=(cm_scx.shape[0], 0), xlabel="Observed {}\n".format(task),
           title="""{}\n{}""".format(task, metric_str), ylabel="Predicted {}".format(task))
    sns.despine()
    return ax


def add_scatter(y, yhat, task, ax, color):  # pragma: no cover
    """Generate a scatter plot to visualize prediction results and return plot to given axes.

    Args:
        y: ar-like, observed
        yhat: ar-like, predicted
        task: ar-like, task names
        ax: axes, matplotlib axes
        color: color, Either named color RGB for usage in matplotlib / seaborn.

    Returns:
        axes, matplotlib axes object
    """
    # get min, max for plotting
    xmin, xmax = np.hstack([y, yhat]).min(), np.hstack([y, yhat]).max()
    xmin = xmin - 0.1 * xmin
    xmax = xmax + 0.1 * xmax
    metric_str = """r2: {:.2f} """.format(custom_r2(y, yhat))
    metric_str = """r2: {:.2f} """.format(custom_r2(y, yhat))
    logger.info("QC: {}".format(task))
    logger.info("Metrics: {}".format(metric_str))
    ax.scatter(y, yhat, facecolor="none", edgecolor=color)
    ax.set(title=metric_str, xlabel="Observed {}".format(task.upper()),
           ylabel="Predicted {}".format(task), xlim=(xmin, xmax), ylim=(xmin, xmax))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    sns.despine()
    return ax


def save_fig(fig, path, outname):  # pragma: no cover
    """Save a figure in png/svg/pdf format and 600 dpi.

    Args:
        fig: figure object, matplotlib figure object
        path: str, path to store the result

    Returns:
        None
    """
    fig.savefig(os.path.join(path, outname + ".png"), dpi=600)
    fig.savefig(os.path.join(path, outname + ".pdf"), dpi=600)
    fig.savefig(os.path.join(path, outname + ".svg"), dpi=600)


def custom_r2(y, yhat):
    """Compute the r2 value.

    Args:
        y: ar-like, observed
        yhat: ar-like, predicted

    Returns:
        float, r2 value
    """
    return np.round(pearsonr(y, yhat)[0] ** 2, 2)


def relaxed_accuracy(y, yhat):
    """Compute the relaxed accuracy (within an error margin of 1).

    Args:
        y:  ar-like, observed values
        yhat:  ar-like, predicted values

    Returns:
        Float, relaxed accuracy (error +- 1
    """
    return np.round(sum(np.abs(y - yhat) <= 1) / len(yhat), 2)


def plot_epoch_cv(callback_path, tasks, xirt_params, outpath, show=False):  # pragma: no cover
    """Do a QC plot of the metrics and losses for all tasks across the epochs (over all cvs).

    Args:
        callback_path: str, path to the callback logs
        tasks: ar-like, list of tasks that were used as identifier in xirt
        xirt_params: dict, parsed yaml file for xirt config.
        outpath: str, location to store the plots
        show: bool, if True plot figure
    Returns:
        None
    """
    # %%    # read epoch log from callbacks
    epochlogs = glob.glob(os.path.join(callback_path, "*epochlog*"))
    df = pd.concat(pd.read_csv(i) for i in epochlogs)

    if len(tasks) == 1:
        # write task as prefix for columns
        df.columns = [tasks[0] + "-" + i if i not in ["epoch", "lr"] else i for i in df.columns]

    # transform and melt dataframe for easier plotting, can probably be optimized
    # RT = retention time
    dfs = []
    for rt_task in tasks:
        df_temp = df.filter(regex="{}|epoch".format(rt_task)).melt(id_vars="epoch")
        df_temp["RT"] = rt_task
        # normalize to max 1
        df_temp["variable_norm"] = df_temp.groupby('variable').transform(lambda x: (x / x.max()))[
            "value"]
        dfs.append(df_temp)

    dfs = pd.concat(dfs)
    dfs["Split"] = ["Training" if "val" not in i else "Validation" for i in dfs["variable"]]
    dfs["Metric"] = ["Loss" if "loss" in i.lower() else "Metric" for i in dfs["variable"]]

    # split loss and metric data
    pattern_frac = "|".join(xirt_params["predictions"]["fractions"])
    pattern_cont = "|".join(xirt_params["predictions"]["continues"])
    nfracs = len(xirt_params["predictions"]["fractions"])
    ncont = len(xirt_params["predictions"]["continues"])
    for filter in ["Metric", "Loss"]:
        if filter == "Metric":
            cname = "metrics"
        elif filter == "Loss":
            cname = "loss"

        # filter for metric/loss
        df_temp = dfs[dfs["Metric"] == filter]

        # split data by fractionation / continuous
        df_temp_frac = df_temp[df_temp.RT.str.contains(pattern_frac)]
        df_temp_cont = df_temp[df_temp.RT.str.contains(pattern_cont)]
        df_temp_cont = df_temp_cont[~df_temp_cont["variable"].str.contains("r_square")]

        # get the metrics and stuff for training, validation (thus the, 2 in tile)
        frac_metrics = sorted(df_temp_frac["variable"].drop_duplicates())
        frac_hues = dict(zip(frac_metrics, np.tile(colors[:nfracs], 2)))

        cont_metrics = sorted(df_temp_cont["variable"].drop_duplicates())
        cont_hues = dict(zip(cont_metrics, np.tile(colors[nfracs:nfracs + ncont], 2)))

        # metric
        f, ax = plt.subplots()
        # make custom legend because lineplot will duplicate the legends with the same color
        if (len(frac_hues) > 0) & (len(cont_hues) > 0):
            handles_frac = [Line2D([0], [0], label=li, color=frac_hues[li]) for li, ci in
                            zip(frac_metrics[:nfracs], frac_hues)]
            handles_cont = [Line2D([0], [0], label=li, color=cont_hues[li]) for li, ci in
                            zip(cont_metrics[:ncont], cont_hues)]

            ax = sns.lineplot(x="epoch", y="value", hue="variable", style="Split",
                              data=df_temp_frac, ax=ax, palette=frac_hues, legend=False)

            ax.set(
                ylabel=xirt_params["output"][
                    xirt_params["predictions"]["fractions"][0] + "-" + cname])
            ax2 = ax.twinx()
            ax2 = sns.lineplot(x="epoch", y="value", hue="variable", style="Split",
                               data=df_temp_cont, ax=ax2, legend=False, palette=cont_hues)
            ax2.set(
                ylabel=xirt_params["output"][
                    xirt_params["predictions"]["continues"][0] + "-" + cname])
            sns.despine(right=False)
            ax.legend(handles=list(np.hstack([handles_frac, handles_cont,
                                              Line2D([0], [0], color="k", ls="-", label="training"),
                                              Line2D([0], [0], color="k", ls="--",
                                                     label="validation")])),
                      loc=2, ncol=3, bbox_to_anchor=(0.01, 1.2), borderaxespad=0.)

            for axt in [ax, ax2]:
                axt.yaxis.set_major_locator(ticker.MaxNLocator(5))

        else:
            # deal with single plot case ...
            hues = cont_hues if len(cont_hues) > 0 else frac_hues
            param_col = "continues" if len(cont_hues) > 0 else "fractions"
            ax = sns.lineplot(x="epoch", y="value", hue="variable", style="Split",
                              data=df_temp_cont, ax=ax, palette=hues, legend="full")
            ax.set(ylabel=xirt_params["output"][
                xirt_params["predictions"][param_col][0] + "-" + cname])
            sns.despine(right=False)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        plt.tight_layout()
        if show:
            plt.show()
        save_fig(f, outpath, outname="cv_epochs_{}".format(cname))
        plt.clf()


def plot_summary_strip(summary_df, tasks, xirt_params, outpath):  # pragma: no cover
    """Plot stripplot to summarize learning results.

    Args:
        summary_df: df, summary dataframe from xirt
        tasks: ar-like, list of tasks
        xirt_params: dict, xirt parameters
        outpath: str, location for storing plots
    Returns:
        None
    """
    # %%
    if len(tasks) == 1:
        colstr = "{}_ordinal-accuracy".format(tasks[0])
        summary_df.columns = [tasks[0] + "_" + i if i not in [colstr, "CV", "Split"] else i for i
                              in summary_df.columns]

    for col in tasks:
        if "ordinal" in xirt_params["output"][col + "-column"]:
            # rewrite params to use accuracy
            xirt_params["output"][col + "-metrics"] = "accuracy"
            # store at right place
            summary_df[col + "_accuracy"] = summary_df[col + "_ordinal-accuracy"].astype(float)

    metrics = [xirt_params["output"][i + "-metrics"] for i in tasks]
    for eval_method in [metrics, "losses"]:
        f, axes = plt.subplots(1, len(tasks), figsize=(5, 4))
        if len(tasks) == 1:
            axes = [axes]

        for ii, m, t in zip(range(len(tasks)), eval_method, tasks):
            if eval_method == "losses":
                m = "loss"
                store_str = "loss"
            else:
                store_str = "metric"
            # points
            axes[ii] = sns.stripplot(x="Split", y="{}_{}".format(t, m), data=summary_df,
                                     ax=axes[ii], color=colors[ii])
            # errors bars
            axes[ii] = sns.pointplot(x="Split", y="{}_{}".format(t, m), data=summary_df, join=True,
                                     color="grey", ax=axes[ii], scale=0.7)
            if m == "accuracy":
                ylim_ax = 1
            else:
                ylim_ax = axes[ii].get_ylim()[1] + 0.2 * axes[ii].get_ylim()[1]

            axes[ii].set(title=t, ylabel=m, xlabel="", ylim=(0, ylim_ax))
            sns.despine(ax=axes[ii])

        # set axes
        for ax in axes:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        plt.tight_layout()
        save_fig(f, outpath, outname="cv_summary_strip_{}".format(store_str))
        plt.clf()
        # %%


def plot_cv_predictions(df_predictions, input_psms, xirt_params, outpath,
                        show=False):  # pragma: no cover
    """Plot observed vs. predicted figure for all tasks.

    Args:
        df_predictions: df, dataframe with predictions and cv annotation
        input_psms: df, input dataframe after xiRT is completed
        xirt_params: dict, xirt parameters
        outpath: str, location to store the figures
        show: bool, if true plots the figure, else just store them

    Returns:
        None
    df_predictions, input_psms = training_data.prediction_df, training_data.psms
                               xirt_params, outpath=xirt_params, outpath
    """
    # %%
    for i in df_predictions["cv"].drop_duplicates():
        # filter data to CV iteration
        temp_cv_df = df_predictions[df_predictions["cv"] == i]
        temp_in_df = input_psms.loc[temp_cv_df.index.values]

        # organize plotting stuff
        fracs = sorted(xirt_params["predictions"]["fractions"])
        cont = sorted(xirt_params["predictions"]["continues"])
        ntasks = len(fracs) + len(cont)

        f, axes = plt.subplots(1, ntasks, figsize=(4 * ntasks, 4))
        if ntasks == 1:
            axes = [axes]

        idx = 0
        # plot heatmap variant for variant for fraction prediction
        for frac_task in fracs:
            dims = np.arange(1, xirt_params["output"][frac_task + "-dimension"] + 1)
            axes[idx] = add_heatmap(temp_in_df[frac_task + "_0based"],
                                    temp_cv_df[frac_task + "-prediction"],
                                    task=frac_task, ax=axes[idx], colormap=colormaps[idx],
                                    dims=dims)
            idx += 1
        # plot scatter variant for continues predictions
        for cont_task in cont:
            axes[idx] = add_scatter(temp_in_df[cont_task].values,
                                    temp_cv_df[cont_task + "-prediction"].values, task=cont_task,
                                    ax=axes[idx], color=colors[idx])
            idx += 1
        axes[int(ntasks / 2)].set(xlabel=axes[int(ntasks / 2)].get_xlabel() + "\nCV: {}".format(i))
        plt.tight_layout()
        if show:
            plt.show()
        save_fig(f, outpath, "qc_cv{}_obs_pred".format(str(i).zfill(2)))
        plt.clf()
    # %%


def plot_classification_report(y, yhat, title, path, colormap=plt.cm.Blues,
                               name=""):  # pragma: no cover
    """Plot a classification report from sklearns API.

    Args:
        y: ar-like, observed
        yhat: ar-like, predicted
        title: str, title
        path: str, location for storing
        colormap: colormap, matplotlib colormap (default: plt.cm.Blues)
        name: str, name for the plot (for storing)

    Returns:
        None
    """
    cr = pd.DataFrame(classification_report(y, yhat, output_dict=True)).transpose()
    f, ax = plt.subplots()
    # annotate the number of observations per row
    cr.index = cr.index + " (" + cr.support.astype(int).astype(str).values + ")"
    # drop this column now for better plotting
    cr = cr.drop("support", axis=1)
    ax = sns.heatmap(cr, cmap=colormap, annot=True, ax=ax)
    ax.set(xlabel="Metrics", ylabel="Classes", title="{} Classification Report".format(title))
    plt.tight_layout()
    save_fig(f, path, "classification_report_" + name)
    plt.clf()


def plot_error_characteristics(df_errors, input_psms, tasks, xirt_params, outpath,
                               plt_func=sns.boxenplot, min_fdr=0.0,
                               max_fdr=0.01):  # pragma: no cover
    """Generate error characteristic boxplot.

    *in development*

    Args:
        df_errors:
        input_psms:
        tasks:
        xirt_params:
        outpath:
        plt_func:
        min_fdr:
        max_fdr:

    Returns:
        None
    """
    # %%
    # fdr filter
    df_all_info = input_psms[(input_psms["fdr"] <= max_fdr) & (input_psms["fdr"] >= min_fdr)]
    # remove duplicates
    df_all_info = df_all_info[~ df_all_info["Duplicate"]]
    # merge dataf rames
    df_all_info = df_all_info.join(df_errors)
    # unify decoy annotation
    ref = np.array(["TT", "TD", "DD"])
    df_all_info["Type"] = ref[np.argmax(df_all_info[["isTT", "isTD", "isDD"]].values, axis=1)]
    # convenient ass to columns
    tasks_errors = [i + "-error" for i in tasks]
    # melt data
    df_ec_melt = df_all_info[np.concatenate([tasks_errors, ["Type"]])].melt(id_vars=["Type"], )
    df_ec_melt = df_ec_melt.rename({"Type": "PSM type"}, axis=1)
    df_ec_melt["variable"] = df_ec_melt["variable"].str.replace("-error", "")
    # counts = dict(df_all_info["isTT"].value_counts())

    # organize plotting stuff
    fracs = xirt_params["predictions"]["fractions"]
    cont = xirt_params["predictions"]["continues"]
    ntasks = len(fracs) + len(cont)

    # %%
    if (len(fracs) > 0) & (len(cont) > 0):
        f, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [len(fracs), len(cont)]})
    else:
        f, axes = plt.subplots(1, ntasks)

    idx = 0
    if len(fracs) > 0:
        df_ec_melt_frac = df_ec_melt[df_ec_melt["variable"].str.contains("|".join(fracs))]
        axes[idx] = plt_func(x="variable", y="value", hue="PSM type", data=df_ec_melt_frac,
                             ax=axes[idx], hue_order=["TT", "TD", "DD"], palette=targetdecoys_cm)
        pairs = [((i, "TT"), (i, "TD")) for i in fracs]
        statannot.add_stat_annotation(axes[idx], data=df_ec_melt_frac,
                                      x="variable", y="value", hue="PSM type", test="t-test_ind",
                                      text_format="star", loc="outside", verbose=2, box_pairs=pairs)
        idx += 1

    if len(cont) > 0:
        df_ec_melt_cont = df_ec_melt[df_ec_melt["variable"].str.contains("|".join(cont))]
        axes[idx] = plt_func(x="variable", y="value", hue="PSM type",
                             data=df_ec_melt_cont, ax=axes[idx],
                             hue_order=["TT", "TD", "DD"], palette=targetdecoys_cm)
        pairs = [((i, "TT"), (i, "TD")) for i in cont]
        statannot.add_stat_annotation(axes[idx], data=df_ec_melt_cont,
                                      x="variable", y="value", hue="PSM type", test="t-test_ind",
                                      text_format="star", loc="outside", verbose=2, box_pairs=pairs)
        idx += 1

    for idx, ax in enumerate(axes):
        ax.axhline(0, lw=1, zorder=0, c="k")
        ax.set(xlabel="", ylabel="Observed - Predicted")
        if idx > 0:
            ax.set(ylabel="")
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        sns.despine(ax=ax)
    save_fig(f, outpath, "error_characteristics_")
    plt.clf()
