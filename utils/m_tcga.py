import scanpy as sc
import os
import anndata
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union


class pyTCGA(object):
    r"""
    TCGA (The Cancer Genome Atlas) data analysis module.

    This class provides comprehensive functionality for downloading, processing,
    and analyzing TCGA genomic and clinical data.
    """

    def __init__(self, gdc_sample_sheep: str, gdc_download_files: str, clinical_cart: str):
        r"""Initialize TCGA analysis module.

        Arguments:
            gdc_sample_sheep: Path to TCGA Sample Sheet TSV file
            gdc_download_files: Path to downloaded TCGA data files directory
            clinical_cart: Path to TCGA clinical data tar.gz file

        """
        self.gdc_sample_sheep = gdc_sample_sheep
        self.gdc_download_files = gdc_download_files
        self.clinical_cart = clinical_cart
        exist_files = [i for i in os.listdir(gdc_download_files) if 'txt' not in i]

        self.sample_sheet = pd.read_csv(self.gdc_sample_sheep, sep='\t', index_col=0)
        exist_files = list(set(exist_files) & set(self.sample_sheet.index))
        self.sample_sheet = self.sample_sheet.loc[exist_files]
        self.clinical_sheet = pd.read_csv('{}/clinical.tsv'.format(self.clinical_cart), sep='\t', index_col=0)
        # self.clinical_sheet=self.clinical_sheet.loc[exist_files]

        sample_index = self.sample_sheet.index[0]
        sample_id = self.sample_sheet.loc[sample_index, 'Sample ID']
        sample_file_id = sample_index
        sample_file_name = self.sample_sheet.loc[sample_index, 'File Name']
        self.data_test = pd.read_csv('{}/{}/{}'.format(self.gdc_download_files, sample_file_id, sample_file_name),
                                     sep='\t', index_col=0, skiprows=1)
        print('tcga module init success')

    def adata_read(self, path: str):
        r"""Read AnnData object from file.

        Arguments:
            path: Path to AnnData file
        """
        print('... anndata reading')
        self.adata = sc.read(path)

    def adata_init(self):
        self.index_init()
        self.expression_init()
        self.matrix_construct()

    def adata_meta_init(self, var_names: list = ['gene_name', 'gene_type'],
                        obs_names: list = ['Case ID', 'Tissue Type']) -> anndata.AnnData:
        """Enhanced metadata initialization with dimension checking."""
        print('...anndata meta init', var_names, obs_names)
        adata = self.adata
        if not hasattr(self, 'data_test'):
            raise AttributeError("data_test attribute not found. Please load gene metadata first.")

        common_genes = adata.var.index.intersection(self.data_test.index)
        missing_in_meta = adata.var.index.difference(self.data_test.index)
        missing_in_adata = self.data_test.index.difference(adata.var.index)

        print(f"Original adata genes: {len(adata.var.index)}")
        print(f"Genes in metadata: {len(self.data_test.index)}")
        print(f"Common genes: {len(common_genes)}")
        print(f"Genes missing in metadata: {len(missing_in_meta)}")
        if len(missing_in_meta) > 0:
            print("First 5 missing genes:", list(missing_in_meta)[:5])

        var_pd = pd.DataFrame(index=adata.var.index)
        var_pd[var_names] = self.data_test.loc[common_genes, var_names]


        if len(missing_in_meta) > 0:
            for col in var_names:
                if col not in var_pd.columns:
                    var_pd[col] = np.nan

        sample_sheet_tmp = self.sample_sheet.copy()
        sample_sheet_tmp.index = sample_sheet_tmp['Sample ID']
        obs_pd = sample_sheet_tmp.loc[adata.obs.index, obs_names]
        obs_pd = obs_pd[~obs_pd.index.duplicated(keep='first')]

        adata.obs = obs_pd.loc[adata.obs.index]
        adata.var = var_pd.loc[adata.var.index]


        if 'gene_name' in var_pd.columns:
            adata.var['gene_id'] = adata.var.index
            adata.var.index = adata.var['gene_name'].astype(str)
            adata.var_names_make_unique()

        print(f"Final adata shape: {adata.shape}")
        print(f"var columns: {adata.var.columns.tolist()}")

        self.adata = adata
        return adata

    def survial_init(self):
        r"""Initialize survival analysis data.

        Processes clinical data to extract survival information including
        vital status and survival days.
        """
        pd_c = self.clinical_sheet

        if isinstance(pd_c['demographic.vital_status'].iloc[0], pd.Series):
            pd_c['demographic.vital_status'] = pd_c['demographic.vital_status'].apply(lambda x: x.iloc[0])


        pd_c['days'] = np.where(
            pd_c['demographic.vital_status'] == 'Dead',
            pd_c['demographic.days_to_death'].apply(lambda x: x.iloc[0] if isinstance(x, pd.Series) else x),
            pd_c['diagnoses.days_to_last_follow_up'].apply(lambda x: x.iloc[0] if isinstance(x, pd.Series) else x)
        )


        pd_c['days'] = pd.to_numeric(pd_c['days'], errors='coerce')
        s_pd = pd_c[["cases.submitter_id",
                     "demographic.vital_status",
                     "diagnoses.days_to_last_follow_up",
                     "demographic.days_to_death",
                     "demographic.age_at_index",
                     "diagnoses.tumor_grade", "days"]].copy()
        s_pd = s_pd.drop_duplicates(subset='cases.submitter_id')
        s_pd.set_index(s_pd.columns[0], inplace=True)
        self.s_pd = s_pd

        self.adata.obs['demographic.vital_status'] = 'Not Reported'
        self.adata.obs['days'] = np.nan
        for i in self.adata.obs.index:
            if self.adata.obs.loc[i, 'Case ID'] not in s_pd.index:
                self.adata = self.adata[self.adata.obs.index != i]
                continue
            self.adata.obs.loc[i, 'demographic.vital_status'] = s_pd.loc[
                self.adata.obs.loc[i, 'Case ID'], 'demographic.vital_status']
            self.adata.obs.loc[i, 'days'] = s_pd.loc[self.adata.obs.loc[i, 'Case ID'], 'days']

    def index_init(self) -> list:
        r"""Initialize gene indices for AnnData construction.

        Returns:
            all_lncRNA_index: List of all gene indices from TCGA samples
        """
        print('...index init')
        all_lncRNA_index = []
        for sample_index in self.sample_sheet.index:
            sample_id = self.sample_sheet.loc[sample_index, 'Sample ID']
            sample_file_id = sample_index
            sample_file_name = self.sample_sheet.loc[sample_index, 'File Name']
            data_test = pd.read_csv('{}/{}/{}'.format(self.gdc_download_files, sample_file_id, sample_file_name),
                                    sep='\t', index_col=0, skiprows=1)
            # data_test=data_test.loc[data_test['gene_type']=='lncRNA']
            data_c_s = data_test['tpm_unstranded'].sort_values(ascending=False)
            data_c_s = data_c_s[~data_c_s.index.duplicated(keep='first')]
            all_lncRNA_index = list(set(all_lncRNA_index) | set(data_c_s.index.tolist()))
        self.tcga_index = all_lncRNA_index
        return all_lncRNA_index

    def expression_init(self):
        r"""Initialize expression matrices for TCGA data.

        Creates count, TPM, and FPKM expression matrices from TCGA files.
        """
        print('... expression matrix init')
        data_pd_count = pd.DataFrame(index=self.tcga_index)
        data_pd_tpm = pd.DataFrame(index=self.tcga_index)
        data_pd_fpkm = pd.DataFrame(index=self.tcga_index)

        for sample_index in self.sample_sheet.index:
            sample_id = self.sample_sheet.loc[sample_index, 'Sample ID']
            sample_file_id = sample_index
            sample_file_name = self.sample_sheet.loc[sample_index, 'File Name']
            # print(sample_id)
            data_test = pd.read_csv('{}/{}/{}'.format(self.gdc_download_files, sample_file_id, sample_file_name),
                                    sep='\t', index_col=0, skiprows=1)
            # data_test=data_test.loc[data_test['gene_type']=='lncRNA']
            data_c_s = data_test['unstranded'].sort_values(ascending=False)
            data_c_s = data_c_s[~data_c_s.index.duplicated(keep='first')]
            data_pd_count[sample_id] = 0
            data_pd_count.loc[data_c_s.index, sample_id] = data_c_s.values

            data_c_s = data_test['tpm_unstranded'].sort_values(ascending=False)
            data_c_s = data_c_s[~data_c_s.index.duplicated(keep='first')]
            data_pd_tpm[sample_id] = 0
            data_pd_tpm.loc[data_c_s.index, sample_id] = data_c_s.values

            data_c_s = data_test['fpkm_unstranded'].sort_values(ascending=False)
            data_c_s = data_c_s[~data_c_s.index.duplicated(keep='first')]
            data_pd_fpkm[sample_id] = 0
            data_pd_fpkm.loc[data_c_s.index, sample_id] = data_c_s.values

        self.data_pd_count = data_pd_count
        self.data_pd_tpm = data_pd_tpm
        self.data_pd_fpkm = data_pd_fpkm
        self.data_test = data_test

    def matrix_construct(self):
        r"""Construct AnnData object from expression matrices.

        Creates AnnData object with multiple layers including raw counts,
        TPM, FPKM, and DESeq2-normalized expression.
        """
        print('...anndata construct')
        var_pd = pd.DataFrame(index=self.data_pd_count.index)
        obs_pd = pd.DataFrame(index=self.data_pd_count.columns)
        adata = anndata.AnnData(self.data_pd_count.T, var=var_pd, obs=obs_pd)
        adata.layers['tpm'] = self.data_pd_tpm.T.values
        adata.layers['fpkm'] = self.data_pd_fpkm.T.values
        adata.layers['deseq_normalize'] = self.matrix_normalize(self.data_pd_count).T.values
        self.adata = adata
        return adata

    def matrix_normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        r"""Normalize expression matrix using DESeq2 method.

        Arguments:
            data: Raw count expression matrix to normalize

        Returns:
            data: DESeq2-normalized expression matrix
        """
        avg1 = data.apply(np.log, axis=1).mean(axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        data1 = data.loc[avg1.index]
        data_log = data1.apply(np.log, axis=1)
        scale = data_log.sub(avg1.values, axis=0).median(axis=0).apply(np.exp)
        return data / scale

    def survival_analysis(self, gene: str, layer: str = 'raw', plot: bool = False, gene_threshold: str = 'median') -> \
            Tuple[float, float]:
        r"""Perform survival analysis for a specific gene."""
        from scipy.sparse import issparse

        goal_gene = gene
        s_pd = self.s_pd.loc[self.adata.obs['Case ID']]

        if layer != 'raw':
            if layer not in self.adata.layers.keys():
                if issparse(self.adata.X):
                    s_pd[goal_gene] = self.adata[self.adata.obs.index, self.adata.var['gene_name'] == goal_gene].X.mean(
                        axis=1).toarray().flatten()
                else:
                    s_pd[goal_gene] = self.adata[self.adata.obs.index, self.adata.var['gene_name'] == goal_gene].X.mean(
                        axis=1)
            else:
                if issparse(self.adata.layers[layer]):
                    s_pd[goal_gene] = self.adata[self.adata.obs.index, self.adata.var['gene_name'] == goal_gene].layers[
                        layer].mean(axis=1).toarray().flatten()
                else:
                    s_pd[goal_gene] = self.adata[self.adata.obs.index, self.adata.var['gene_name'] == goal_gene].layers[
                        layer].mean(axis=1)
        else:
            if issparse(self.adata.X):
                s_pd[goal_gene] = self.adata[self.adata.obs.index, self.adata.var['gene_name'] == goal_gene].X.mean(
                    axis=1).toarray().flatten()
            else:
                s_pd[goal_gene] = self.adata[self.adata.obs.index, self.adata.var['gene_name'] == goal_gene].X.mean(
                    axis=1)


        s_pd = s_pd.dropna(subset=[goal_gene, 'days', 'demographic.vital_status'])


        if gene_threshold == 'median':
            s_pd[f'{goal_gene}_status'] = np.where(s_pd[goal_gene] > s_pd[goal_gene].median(), 'High', 'Low')
        elif gene_threshold == 'mean':
            s_pd[f'{goal_gene}_status'] = np.where(s_pd[goal_gene] > s_pd[goal_gene].mean(), 'High', 'Low')
        else:
            s_pd[f'{goal_gene}_status'] = np.where(s_pd[goal_gene] > gene_threshold, 'High', 'Low')

        s_pd = s_pd[s_pd['days'] != "'--"]
        s_pd['days'] = pd.to_numeric(s_pd['days'], errors='coerce')
        s_pd = s_pd.dropna(subset=['days'])

        s_pd['fustat'] = np.where(s_pd['demographic.vital_status'] == 'Alive', 0, 1)

        T = s_pd['days'].astype(float) / 365
        E = s_pd['fustat']
        gender = (s_pd[f'{goal_gene}_status'] == 'High')


        valid_idx = ~(T.isna() | E.isna() | gender.isna())
        T = T[valid_idx]
        E = E[valid_idx]
        gender = gender[valid_idx]
        print(f'high_cout:{sum(gender)}, low_count:{sum(~gender)}')

        lr = logrank_test(T[gender], T[~gender], E[gender], E[~gender], alpha=.95)
        from lifelines import CoxPHFitter
        try:
            cox_data = pd.DataFrame({
                'time': pd.to_numeric(s_pd['days'], errors='coerce'),
                'event': pd.to_numeric(s_pd['fustat'], errors='coerce'),
                'group': (s_pd[f'{goal_gene}_status'] == 'High').astype(int)
            }).dropna()


            if len(cox_data) < 10:
                raise ValueError(f"Only {len(cox_data)} samples available (min 10 required)")
            if cox_data['event'].sum() < 5:
                raise ValueError(f"Only {cox_data['event'].sum()} events available (min 5 required)")
            if len(cox_data['group'].unique()) < 2:
                raise ValueError("Only one group available (both High and Low required)")

            # 拟合Cox模型            cph = CoxPHFitter()
            cph.fit(cox_data, duration_col='time', event_col='event')


            summary_df = cph.summary
            hr = summary_df.loc['group', 'exp(coef)']
            hr_ci_lower = summary_df.loc['group', 'exp(coef) lower 95%']
            hr_ci_upper = summary_df.loc['group', 'exp(coef) upper 95%']
            hr_pvalue = summary_df.loc['group', 'p']

            print(f"\nHazard Ratio (HR) Analysis for {goal_gene}:")
            print(f"HR (High vs Low): {hr:.3f} (95% CI: {hr_ci_lower:.3f}-{hr_ci_upper:.3f})")
            print(f"P-value: {hr_pvalue:.4f}")

        except Exception as e:
            print(f"\nFailed to calculate HR for {goal_gene}: {str(e)}")
            hr = np.nan
            hr_ci_lower = np.nan
            hr_ci_upper = np.nan
            hr_pvalue = np.nan
        if plot:

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            time_ranges = [(None, None, "Full Range"),
                           (0, 5, "0-5 Years"),
                           (0, 10, "0-10 Years")]


            high_label = f"High (n={sum(gender)})"
            low_label = f"Low (n={sum(~gender)})"

            for ax, (start, end, title_text) in zip(axes, time_ranges):
                km = KaplanMeierFitter()


                if start is not None:
                    mask = (T >= start) & (T <= end)
                    T_subset = T[mask]
                    E_subset = E[mask]
                    gender_subset = gender[mask]
                else:
                    T_subset = T
                    E_subset = E
                    gender_subset = gender


                lr = logrank_test(T_subset[gender_subset],
                                  T_subset[~gender_subset],
                                  E_subset[gender_subset],
                                  E_subset[~gender_subset])

                km.fit(T_subset[gender_subset],
                       event_observed=E_subset[gender_subset],
                       label=high_label)
                km.plot(ax=ax, color='#941456')


                km.fit(T_subset[~gender_subset],
                       event_observed=E_subset[~gender_subset],
                       label=low_label)
                km.plot(ax=ax, color='#368650')


                if start is not None:
                    ax.set_xlim(start, end)

                current_high = sum(gender_subset)
                current_low = sum(~gender_subset)
                info_text = f"High: {current_high}\nLow: {current_low}\n"
                info_text += f"p = {lr.p_value:.3f}" if lr.p_value >= 0.001 else "p < 0.001"

                ax.text(1, 1, info_text, transform=ax.transAxes,
                        bbox=dict(facecolor='white', alpha=0.8))


                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_xlabel('Years')
                ax.set_ylabel('Survival Probability')
                ax.set_title(f'{goal_gene}\n{title_text}')
                ax.grid(False)

            plt.tight_layout()

        return lr.test_statistic, lr.p_value

    # def survival_analysis(self,gene:str,layer:str='raw',plot:bool=False,gene_threshold:str='median')->Tuple[float,float]:
    #     r"""Perform survival analysis for a specific gene.
    #
    #     Arguments:
    #         gene: Gene name for survival analysis
    #         layer: AnnData layer to use for expression values (default: 'raw')
    #         plot: Whether to generate Kaplan-Meier survival plot (default: False)
    #         gene_threshold: Method to split samples into high/low expression groups
    #                       (default: 'median', options: 'median', 'mean', or numeric value)
    #
    #     Returns:
    #         test_statistic: Log-rank test statistic
    #         pvalue: Log-rank test p-value
    #
    #     """
    #     from scipy.sparse import issparse
    #     goal_gene=gene
    #
    #     s_pd=self.s_pd
    #     s_pd=s_pd.loc[self.adata.obs['Case ID']]
    #     if layer!='raw':
    #         if layer not in self.adata.layers.keys():
    #             #issparse
    #
    #             if issparse(self.adata.X):
    #                 s_pd[goal_gene]=self.adata[self.adata.obs.index,self.adata.var['gene_name']==goal_gene].X.mean(axis=1).toarray()
    #             else:
    #                 s_pd[goal_gene]=self.adata[self.adata.obs.index,self.adata.var['gene_name']==goal_gene].X.mean(axis=1)
    #         else:
    #
    #             if issparse(self.adata.layers[layer]):
    #                 s_pd[goal_gene]=self.adata[self.adata.obs.index,self.adata.var['gene_name']==goal_gene].layers[layer].mean(axis=1).toarray()
    #             else:
    #                 s_pd[goal_gene]=self.adata[self.adata.obs.index,self.adata.var['gene_name']==goal_gene].layers[layer].mean(axis=1)
    #
    #     else:
    #         if issparse(self.adata.X):
    #             s_pd[goal_gene]=self.adata[self.adata.obs.index,self.adata.var['gene_name']==goal_gene].X.mean(axis=1).toarray()
    #         else:
    #             s_pd[goal_gene]=self.adata[self.adata.obs.index,self.adata.var['gene_name']==goal_gene].X.mean(axis=1)
    #     if gene_threshold=='median':
    #         s_pd['{}_status'.format(goal_gene)]=['High' if i>s_pd[goal_gene].median() else 'Low' for i in s_pd[goal_gene] ]
    #     elif gene_threshold=='mean':
    #         s_pd['{}_status'.format(goal_gene)]=['High' if i>s_pd[goal_gene].mean() else 'Low' for i in s_pd[goal_gene] ]
    #     else:
    #         s_pd['{}_status'.format(goal_gene)]=['High' if i>gene_threshold else 'Low' for i in s_pd[goal_gene] ]
    #     s_pd=s_pd.loc[s_pd['days']!="'--"]
    #     s_pd['fustat'] = [0 if 'Alive'==i else 1 for i in s_pd['vital_status']]
    #     s_pd['gene_fustat'] = [0 if 'High'==i else 1 for i in s_pd['{}_status'.format(goal_gene)]]
    #
    #     km = KaplanMeierFitter()
    #     T = s_pd['days'].astype(float) / 365
    #     E=s_pd['fustat']
    #
    #     gender = (s_pd['{}_status'.format(goal_gene)] == 'High')
    #     lr = logrank_test(T[gender], T[~gender], E[gender], E[~gender], alpha=.95)
    #     if plot==True:
    #         fig, ax = plt.subplots(figsize=(3,3))
    #         km.fit(T[gender], event_observed=E[gender], label="High")
    #         km.plot(ax=ax,color='#941456')
    #         km.fit(T[~gender], event_observed=E[~gender], label="Low")
    #         km.plot(ax=ax,color='#368650')
    #         lr = logrank_test(T[gender], T[~gender], E[gender], E[~gender], alpha=.95)
    #         lr.p_value
    #
    #         ax.spines['top'].set_visible(False)
    #         ax.spines['right'].set_visible(False)
    #         ax.spines['bottom'].set_visible(True)
    #         ax.spines['left'].set_visible(True)
    #
    #         plt.xlabel('Years')
    #         plt.ylabel('Pecent Survial')
    #         plt.title('Survial: {}\np-value: {}'.format(goal_gene,round(lr.p_value,3)))
    #         plt.grid(False)
    #
    #     return lr.test_statistic,lr.p_value

    def survial_analysis_all(self):
        r"""Perform survival analysis for all genes in the dataset.

        Calculates survival statistics for every gene and stores results
        in AnnData.var as 'survial_test_statistic' and 'survial_p' columns.
        """
        from tqdm import tqdm
        res_l_lnc = []
        res_l_tt = []
        for i in tqdm(self.adata.var.index):
            res_l_tt.append(self.survival_analysis(i)[0])
            res_l_lnc.append(self.survival_analysis(i)[1])
        self.adata.var['survial_test_statistic'] = res_l_tt
        self.adata.var['survial_p'] = res_l_lnc


