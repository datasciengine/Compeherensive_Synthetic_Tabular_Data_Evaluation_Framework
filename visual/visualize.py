import os
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from shutil import rmtree
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder


class Visualizer:
    def __init__(self, project_name="my_project"):
        self.project_name = project_name

    def get_dir(self, func_name):
        if not os.path.isdir(f"eval_results/{self.project_name}/"):
            os.mkdir(f"eval_results/{self.project_name}/")

        if not os.path.isdir(f"eval_results/{self.project_name}/imgs/"):
            os.mkdir(f"eval_results/{self.project_name}/imgs/")

        if not os.path.isdir(f"eval_results/{self.project_name}/imgs/{func_name}/"):
            os.mkdir(f"eval_results/{self.project_name}/imgs/{func_name}/")

        # rmtree(f"eval_results/{self.project_name}/imgs/{func_name}/")
        # os.mkdir(f"eval_results/{self.project_name}/imgs/{func_name}/")

    def univariate_num(self, real_df, syn_df, nums):
        func_name = "univariate_num"
        self.get_dir(func_name)
        for num in nums:
            sns.scatterplot(real_df[num], color="blue", alpha=.75, label="real")
            sns.scatterplot(syn_df[num], color="red", alpha=.75, label="syn")
            plt.savefig(f"eval_results/{self.project_name}/imgs/{func_name}/{num}.png")
            plt.clf()

    def univariate_cat(self, real_df, syn_df, cats):
        func_name = "univariate_cat"
        self.get_dir(func_name)
        for cat in cats:
            sns.countplot(real_df[cat], color="blue", alpha=.5, label="real")
            sns.countplot(syn_df[cat], color="red", alpha=.5, label="syn")
            plt.savefig(f"eval_results/{self.project_name}/imgs/{func_name}/{cat}.png")
            plt.clf()

    def reduced_dimension(self, real_df, syn_df, cats):
        func_name = "reduced_dimension"
        self.get_dir(func_name)
        algorithm = "PCA"
        le = LabelEncoder()

        for cat in cats:
            syn_df[cat] = le.fit_transform(syn_df[cat])
            real_df[cat] = le.fit_transform(real_df[cat])

        # Özellikleri ölçekle
        scaler = StandardScaler()
        syn_df = scaler.fit_transform(syn_df)
        real_df = scaler.fit_transform(real_df)

        pca = PCA(n_components=2)
        syn_pca_df = pd.DataFrame(pca.fit_transform(syn_df))
        real_pca_df = pd.DataFrame(pca.fit_transform(real_df))

        sns.scatterplot(syn_pca_df[0], color="blue", alpha=.75, label="syn")
        sns.scatterplot(syn_pca_df[1], color="blue", alpha=.75)
        sns.scatterplot(real_pca_df[0], color="red", alpha=.75, label="real")
        sns.scatterplot(real_pca_df[1], color="red", alpha=.75)

        plt.savefig(f"eval_results/{self.project_name}/imgs/{func_name}/{algorithm}.png")
        plt.clf()

    def bivariate_num(self, real_df, syn_df, nums):
        func_name = "bivariate_num"
        self.get_dir(func_name)
        for comb in list(itertools.combinations(nums, 2)):
            plt.figure(figsize=(10, 6))
            sns.regplot(data=real_df, x=comb[0], y=comb[1],
                        line_kws={'color': 'blue'}, scatter_kws={'color': 'blue'}, label="real")
            sns.regplot(data=syn_df, x=comb[0], y=comb[1],
                        line_kws={'color': 'red'}, scatter_kws={'color': 'red'}, label="syn")
            plt.legend()
            plt.savefig(f"eval_results/{self.project_name}/imgs/{func_name}/{comb[0]}_{comb[1]}.png")
            plt.clf()

    def bivariate_cat(self, real_df, syn_df, cats):
        func_name = "bivariate_cat"
        self.get_dir(func_name)
        real_df['type'] = 'real'
        syn_df['type'] = 'synthetic'
        combined_df = pd.concat([real_df, syn_df])

        for comb in list(itertools.combinations(cats, 2)):
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=comb[0], y=comb[1], hue="type", data=combined_df)
            plt.legend()
            plt.savefig(f"eval_results/{self.project_name}/imgs/{func_name}/{comb[0]}_{comb[1]}.png")
            plt.clf()

    def plot_corr(self, vis_corr, file_name):
        func_name = "correlation"
        # vis_corr = real_df[nums].corr()
        plt.figure(figsize=(16, 10))
        mask = np.triu(np.ones_like(vis_corr))
        heatmap = sns.heatmap(vis_corr, mask=mask, vmin=-1, vmax=1,
                              annot=True, cmap='BrBG')
        heatmap.set_title(f"{file_name} - Correlation Heatmap", fontdict={'fontsize': 20}, pad=16)
        plt.savefig(f"eval_results/{self.project_name}/imgs/{func_name}/{file_name}.png")
        plt.clf()

    def correlation_graphs(self, real_df, syn_df, nums):
        self.get_dir("correlation")
        real_corr = round(real_df[nums].corr(), 3)
        syn_corr = round(syn_df[nums].corr(), 3)
        diff_corr = round(real_df[nums].corr() - syn_df[nums].corr(), 3)
        self.plot_corr(real_corr, "real")
        self.plot_corr(syn_corr, "syn")
        self.plot_corr(diff_corr, "diff")
