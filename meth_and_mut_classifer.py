import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Function to split data into train, validation, and test sets
def split_data(df_known, rand_i):
    # First, split into 80% (train+validation) and 20% (test)
    df_known_temp, df_known_test = train_test_split(
        df_known, test_size=0.2, stratify=df_known['Label'], random_state=rand_i)
    # Then, split the 80% into 50% train and 50% validation (i.e., 40%/40% of the original)
    df_known_train, df_known_validation = train_test_split(
        df_known_temp, test_size=0.5, stratify=df_known_temp['Label'], random_state=rand_i)
    return df_known_train, df_known_validation, df_known_test

def generate_mutation_features(file_path: str, output_csv: str, genes_file: str = "100_genes.csv"):
    # Load mutation data from CSV
    df = pd.read_csv(file_path)

    # Load the list of valid genes
    valid_genes = pd.read_csv(genes_file)["gene"].tolist()
    filtered_df = df[df["Gene_name"].isin(valid_genes)]

    # Define a set of known driver genes
    driver_genes = {'TP53', 'BRCA1', 'BRCA2', 'KRAS', 'PIK3CA', 'PTEN', 'APC', 'ATM'}

    # Calculate the proportion of mutations in known driver genes (among the 100 selected genes)
    driver_mut_count = filtered_df[filtered_df["Gene_name"].isin(driver_genes)].groupby("case_id").size()
    total_mut_count = filtered_df.groupby("case_id").size()
    prop_driver_mut = (driver_mut_count / total_mut_count).rename("prop_driver_mutations").fillna(0)

    # Create a binary feature indicating presence of a high-impact TP53 mutation
    tp53_high_impact = filtered_df[
        (filtered_df["Gene_name"] == "TP53") &
        (filtered_df["Variant_Classification"].isin(["Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins"]))
    ].groupby("case_id").size()
    tp53_high_impact = tp53_high_impact.apply(lambda x: 1).reindex(filtered_df["case_id"].unique(), fill_value=0).rename("TP53_high_impact")

    # Create a binary feature indicating presence of a high-impact MUC16 mutation
    muc16_high_impact = filtered_df[
        (filtered_df["Gene_name"] == "MUC16") &
        (filtered_df["Variant_Classification"].isin(["Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins"]))
    ].groupby("case_id").size()
    muc16_high_impact = muc16_high_impact.apply(lambda x: 1).reindex(filtered_df["case_id"].unique(), fill_value=0).rename("MUC16_high_impact")

    # Create a binary feature indicating presence of a high-impact MUC4 mutation
    muc4_high_impact = filtered_df[
        (filtered_df["Gene_name"] == "MUC4") &
        (filtered_df["Variant_Classification"].isin(["Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins"]))
    ].groupby("case_id").size()
    muc4_high_impact = muc4_high_impact.apply(lambda x: 1).reindex(filtered_df["case_id"].unique(), fill_value=0).rename("MUC4_high_impact")

    # For AK2 and KMT2C: count distinct Variant_Classification per patient, then binary if >1
    def multi_variant_flag(gene_name, df_in):
        grouped = (
            df_in[df_in["Gene_name"] == gene_name]
            .groupby(["case_id"])["Variant_Classification"]
            .nunique()
            .rename(f"{gene_name}_distinct_variant_count")
        )
        binary = grouped.apply(lambda cnt: 1 if cnt > 1 else 0).reindex(filtered_df["case_id"].unique(), fill_value=0)
        return binary.rename(f"{gene_name}_multiple_variant_types")

    ak2_multitype = multi_variant_flag("AK2", filtered_df)
    kmt2c_multitype = multi_variant_flag("KMT2C", filtered_df)

    #8: Calculate Shannon entropy of mutations across genes for each patient (diversity measure)
    gene_mut_counts = filtered_df.groupby(["case_id", "Gene_name"]).size().unstack(fill_value=0)
    gene_entropy = gene_mut_counts.apply(lambda row: entropy(row.values), axis=1).rename("gene_mutation_entropy")

    # A1-Total number of mutations per patient (all genes)
    total_mut = df.groupby("case_id").size().rename("total_mutations")

    # A2- Count of mutations per type (Variant_Classification) per patient (all genes)
    mut_type_counts = (
        df.groupby(["case_id", "Variant_Classification"])
        .size()
        .unstack(fill_value=0)
        .add_prefix("var_count_")
    )

    # A3- Count of mutations per combination of gene and mutation type per patient (100 genes only)
    gene_type_counts = (
        filtered_df.groupby(["case_id", "Gene_name", "Variant_Classification"])
        .size()
        .reset_index(name="count")
    )
    gene_type_pivot = gene_type_counts.pivot_table(
        index="case_id",
        columns=["Gene_name", "Variant_Classification"],
        values="count",
        fill_value=0
    )
    gene_type_pivot.columns = [
        f"combi_{gene}_{mtype}" for gene, mtype in gene_type_pivot.columns.to_list()
    ]

    #9: Standard deviation and variance of mutation counts per gene (among the 100 genes)
    gene_counts = filtered_df.groupby(["case_id", "Gene_name"]).size().unstack(fill_value=0)
    gene_std = gene_counts.std(axis=1).rename("std_mutations_per_gene")
    gene_var = gene_counts.var(axis=1).rename("var_mutations_per_gene")

    #10: Standard deviation and variance of mutation counts per variant type (among the 100 genes)
    variant_counts = filtered_df.groupby(["case_id", "Variant_Classification"]).size().unstack(fill_value=0)
    variant_std = variant_counts.std(axis=1).rename("std_mutations_per_variant")
    variant_var = variant_counts.var(axis=1).rename("var_mutations_per_variant")

    #11: Number of mutations per gene (among the 100 genes)
    mut_per_gene = filtered_df.groupby(['case_id', 'Gene_name']).size().unstack(fill_value=0).add_prefix('mut_count_gene_')

    #12: Number of mutations per chromosome (among the 100 genes)
    mut_per_chr = filtered_df.groupby(['case_id', 'Chromosome']).size().unstack(fill_value=0).add_prefix('mut_count_chr_')

    #13: Number of unique mutated genes per patient (among the 100 genes)
    num_mutated_genes = filtered_df.groupby("case_id")["Gene_name"].nunique().rename("num_mutated_genes")

    # Count how many gene‐variant combinations have more than one mutation per patient
    gene_variant_counts = (
        filtered_df.groupby(['case_id', 'Gene_name', 'Variant_Classification'])
        .size()
        .reset_index(name='count')
    )
    gene_variant_counts['multiple_mutations'] = (gene_variant_counts['count'] > 1).astype(int)
    gene_var_multiple_mutations = (
        gene_variant_counts.groupby('case_id')['multiple_mutations']
        .sum()
        .rename("gene_var_multiple_mutations")
    )

    #14: Proportion of deletion mutations out of all ins/del mutations
    del_mask = filtered_df['Variant_Classification'].str.contains('del', case=False, na=False)
    ins_mask = filtered_df['Variant_Classification'].str.contains('ins', case=False, na=False)
    del_or_ins_mask = del_mask | ins_mask

    del_counts = filtered_df[del_mask].groupby('case_id').size()
    delins_counts = filtered_df[del_or_ins_mask].groupby('case_id').size()
    del_of_delins_percentage = (del_counts / delins_counts).fillna(0).rename('del_of_delins_percentage')

    # Calculate ratio of reciprocal alleles (allele2 vs allele1)
    def is_reason(row):
        a1 = row['Tumor_Seq_Allele1']
        a2 = row['Tumor_Seq_Allele2']
        reciprocal = {'A': 'G', 'G': 'A', 'T': 'C', 'C': 'T'}
        return 1 if a2 == reciprocal.get(a1, None) else 0

    filtered_df['Reason'] = filtered_df.apply(is_reason, axis=1)
    mismatch_counts = filtered_df.groupby('case_id')['Reason'].sum()
    total_counts = filtered_df.groupby('case_id').size()
    mismatch_ratio = (mismatch_counts / total_counts).fillna(0).rename('reason_ratio')

    #15: C>A Transversion Ratio – known to be enriched in smoking-related cancer (e.g., LUSC)
    def is_CA_transversion(row):
        return int(row['Reference_Allele'] == 'C' and row['Tumor_Seq_Allele2'] == 'A')

    filtered_df['CA_transversion'] = filtered_df.apply(is_CA_transversion, axis=1)
    ca_counts = filtered_df.groupby('case_id')['CA_transversion'].sum()
    total_counts = filtered_df.groupby('case_id').size()
    ca_ratio = (ca_counts / total_counts).fillna(0).rename("CA_transversion_ratio")

    # 16: G>T Transversion Ratio
    def is_GT_transversion(row):
        return int(row['Reference_Allele'] == 'G' and row['Tumor_Seq_Allele2'] == 'T')

    filtered_df['GT_transversion'] = filtered_df.apply(is_GT_transversion, axis=1)
    gt_counts = filtered_df.groupby('case_id')['GT_transversion'].sum()
    total_counts = filtered_df.groupby('case_id').size()
    gt_ratio = (gt_counts / total_counts).fillna(0).rename("GT_transversion_ratio")

    # 17: A>T Transversion Ratio
    def is_AT_transversion(row):
        return int(row['Reference_Allele'] == 'A' and row['Tumor_Seq_Allele2'] == 'T')

    filtered_df['AT_transversion'] = filtered_df.apply(is_AT_transversion, axis=1)
    at_counts = filtered_df.groupby('case_id')['AT_transversion'].sum()
    total_counts = filtered_df.groupby('case_id').size()
    at_ratio = (at_counts / total_counts).fillna(0).rename("AT_transversion_ratio")

    # If "Label" column exists, create a mapping from case_id to Label
    if "Label" in df.columns:
        labels = df[["case_id", "Label"]].drop_duplicates().set_index("case_id")["Label"]

    # Combine all feature series/dataframes into one DataFrame
    features = pd.concat([
        total_mut,
        mut_type_counts,
        gene_type_pivot,
        gene_std,
        gene_var,
        variant_std,
        variant_var,
        mut_per_gene,
        mut_per_chr,
        num_mutated_genes,
        prop_driver_mut,
        tp53_high_impact,
        muc16_high_impact,
        muc4_high_impact,
        ak2_multitype,
        kmt2c_multitype,
        gene_entropy,
        gene_var_multiple_mutations,
        del_of_delins_percentage,
        mismatch_ratio,
        ca_ratio,
        gt_ratio,
        at_ratio
    ], axis=1).fillna(0).reset_index()

    if "Label" in df.columns:
        features["Label"] = features["case_id"].map(labels)

    # Round selected numeric columns for better readability
    round_cols = [
        "std_mutations_per_gene",
        "var_mutations_per_gene",
        "std_mutations_per_variant",
        "var_mutations_per_variant",
        "gene_mutation_entropy",
        "del_of_delins_percentage",
        "homozygous_mutation_ratio"
    ]
    for col in round_cols:
        if col in features.columns:
            features[col] = features[col].round(3)

    features.to_csv(output_csv, index=False)
    print("Mutation features file saved:", output_csv)
    return features

def generate_methylation_features(file_path2: str, output_csv2: str, genes_file: str = "100_genes.csv"):
    upstream = 50
    downstream = 50
    hyper_threshold = 0.8
    hypo_threshold = 0.2

    # Load methylation data from CSV
    df2 = pd.read_csv(file_path2)

    # Load the list of valid genes
    valid_genes = pd.read_csv(genes_file)["gene"].tolist()
    filtered_meth_df = df2[df2["matching_genes"].isin(valid_genes)].copy()

    # Define a set of known driver genes
    driver_genes = {'TP53', 'BRCA1', 'BRCA2', 'KRAS', 'PIK3CA', 'PTEN', 'APC', 'ATM'}

    # Calculate the proportion of methylations in known driver genes (among the 100 selected genes)
    driver_meth_count = filtered_meth_df[filtered_meth_df["matching_genes"].isin(driver_genes)].groupby("case_id").size()

    total_meth_count = filtered_meth_df.groupby("case_id").size()
    prop_driver_meth = (driver_meth_count / total_meth_count).rename("prop_driver_methylations").fillna(0)

    # Feature 1:  mean methylation value per probe per gene
    meth_mean_per_gene = (filtered_meth_df.groupby(["case_id", "matching_genes"])["beta_val"].mean().unstack(fill_value=0).add_prefix("meth_mean_gene_"))

    # Feature 2: ompute CpG_count in the flanking region around each probe
    # (requires a dict of chromosome sequences to function; otherwise CpG_count = 0)
    # If you have chromosome FASTA sequences, load into chrom_sequences dict
    chrom_sequences = None
    if chrom_sequences is not None:
        def compute_cpg(row):
            chrom = str(row["Chromosome"])
            strand = row.get("Strand", "+")
            start = int(row["Start"])
            end = int(row["End"])
            seq_str = chrom_sequences.get(chrom, "")
            if not seq_str:
                return 0
            region = extract_flanked_region(seq_str, strand, start, end, upstream, downstream)
            return region.count("CG")
        filtered_meth_df["CpG_count"] = filtered_meth_df.apply(compute_cpg, axis=1)
    else:
        filtered_meth_df["CpG_count"] = 0
    # Convert to a feature per patient: mean CpG_count across all probes
    avg_cpg_per_case = (filtered_meth_df.groupby("case_id")["CpG_count"].mean().rename("avg_CpG_count"))

    # Feature 3: standard deviation of methylation values per gene
    meth_std_per_gene = (filtered_meth_df.groupby(["case_id", "matching_genes"])["beta_val"].std().unstack(fill_value=0).add_prefix("meth_std_gene_"))

    # Feature 4: unique probe count per gene per patient
    meth_probe_counts = (filtered_meth_df.groupby(["case_id", "matching_genes"])["probeID"].nunique().unstack(fill_value=0).add_prefix("meth_probe_count_gene_"))

    # Feature 5: Shannon entropy of mean methylation between genes
    gene_meth_values = (filtered_meth_df.groupby(["case_id", "matching_genes"])["beta_val"].mean().unstack(fill_value=0))
    gene_meth_entropy = (gene_meth_values.apply(lambda row: entropy(row.values), axis=1).rename("gene_meth_entropy"))

    # Feature 6: proportion of hyper-methylated probes (>= hyper_threshold)
    is_hyper = filtered_meth_df["beta_val"] >= hyper_threshold
    hyper_counts = filtered_meth_df[is_hyper].groupby("case_id").size()
    prop_hyper_meth = ((hyper_counts / total_meth_count).rename("prop_hyper_methylations").fillna(0))

    # Feature 7: proportion of hypo-methylated probes (<= hypo_threshold)
    is_hypo = filtered_meth_df["beta_val"] <= hypo_threshold
    hypo_counts = filtered_meth_df[is_hypo].groupby("case_id").size()
    prop_hypo_meth = ((hypo_counts / total_meth_count).rename("prop_hypo_methylations").fillna(0))

    # If "Label" column exists, create a mapping from case_id to Label
    if "Label" in df2.columns:
        labels2 = df2[["case_id", "Label"]].drop_duplicates().set_index("case_id")["Label"]

    # Combine all feature series/dataframes into one DataFrame
    features2 = pd.concat([
        prop_driver_meth,
        meth_mean_per_gene,
        avg_cpg_per_case,
        meth_std_per_gene,
        meth_probe_counts,
        gene_meth_entropy,
        prop_hyper_meth,
        prop_hypo_meth
    ], axis=1).fillna(0).reset_index()

    if "Label" in df2.columns:
        features2["Label"] = features2["case_id"].map(labels2)

    # Round selected numeric columns for better readability
    round_cols2 = [
        "gene_meth_entropy",
        "prop_driver_methylations",
        "prop_hyper_methylations",
        "prop_hypo_methylations"
    ]
    for col in round_cols2:
        if col in features2.columns:
            features2[col] = features2[col].round(3)

    features2.to_csv(output_csv2, index=False)
    print("Methylation features file saved:", output_csv2)
    return features2

def greedy_feature_selection(df_known_train, df_known_validation, df_known_test, tolerance=0.05, random_state=42):
    feature_location = {
        # All existing mutation groups …
        "Total_Mutations": [col for col in df_known_train.columns if col == "total_mutations"],
        "Mutations_Per_Variant": [col for col in df_known_train.columns if col.startswith("var_count")],
        "Combi_Gene_Var": [col for col in df_known_train.columns if col.startswith("combi")],
        "STD_Gene": [col for col in df_known_train.columns if col == "std_mutations_per_gene"],
        "VAR_Gene": [col for col in df_known_train.columns if col == "var_mutations_per_gene"],
        "STD_Variant": [col for col in df_known_train.columns if col == "std_mutations_per_variant"],
        "VAR_Variant": [col for col in df_known_train.columns if col == "var_mutations_per_variant"],
        "Mutations_Per_Gene": [col for col in df_known_train.columns if col.startswith("mut_count_gene")],
        "Mutations_Per_Chromosome": [col for col in df_known_train.columns if col.startswith("mut_count_chr")],
        "Mutated_Genes": [col for col in df_known_train.columns if col == "num_mutated_genes"],
        "Driver_Genes": [col for col in df_known_train.columns if col == "prop_driver_mutations"],
        "TP53_Binary": [col for col in df_known_train.columns if col == "TP53_high_impact"],
        "MUC16_Binary": [col for col in df_known_train.columns if col == "MUC16_high_impact"],
        "MUC4_Binary": [col for col in df_known_train.columns if col == "MUC4_high_impact"],
        "AK2_MultiType": [col for col in df_known_train.columns if col == "AK2_multiple_variant_types"],
        "KMT2C_MultiType": [col for col in df_known_train.columns if col == "KMT2C_multiple_variant_types"],
        "Entropy": [col for col in df_known_train.columns if col == "gene_mutation_entropy"],
        "Combi_With_Mult": [col for col in df_known_train.columns if col == "gene_var_multiple_mutations"],
        "Del_To_Ins+Del": [col for col in df_known_train.columns if col == "del_of_delins_percentage"],
        "Reason": [col for col in df_known_train.columns if col == "reason_ratio"],
        "CA_Transversion": [col for col in df_known_train.columns if col == "CA_transversion_ratio"],
        # Additional methylation groups
        "Driver_Meth": [col for col in df_known_train.columns if col == "prop_driver_methylations"],
        "Meth_Mean_Per_Gene": [col for col in df_known_train.columns if col.startswith("meth_mean_gene_")],
        "Avg_CpG_Count": [col for col in df_known_train.columns if col == "avg_CpG_count"],
        "Meth_STD_Gene": [col for col in df_known_train.columns if col.startswith("meth_std_gene_")],
        "Meth_Probe_Count": [col for col in df_known_train.columns if col.startswith("meth_probe_count_gene_")],
        "Meth_Entropy": [col for col in df_known_train.columns if col == "gene_meth_entropy"],
        "Hyper_Meth": [col for col in df_known_train.columns if col == "prop_hyper_methylations"],
        "Hypo_Meth": [col for col in df_known_train.columns if col == "prop_hypo_methylations"]
    }

    feature_groups = list(feature_location.keys())
    selected_groups = []
    best_val_error = 1.0
    scaler = StandardScaler()
    improved = True

    y_train = df_known_train['Label']
    y_val = df_known_validation['Label']
    y_test = df_known_test['Label']

    # Greedy feature selection loop
    while improved:
        improved = False
        best_group = None
        for group in feature_groups:
            if group in selected_groups:
                continue
            current_cols = []
            for g in selected_groups + [group]:
                current_cols.extend(feature_location[g])

            X_train = df_known_train[current_cols]
            X_val = df_known_validation[current_cols]

            rf = RandomForestClassifier(random_state=random_state)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            val_error = sum(y_pred != y_val) / len(y_val)

            if val_error < best_val_error:
                best_val_error = val_error
                best_group = group
                improved = True

        if improved and best_group is not None:
            selected_groups.append(best_group)

    # Train final model on the selected features
    final_cols = []
    for g in selected_groups:
        final_cols.extend(feature_location[g])

    X_train_final = df_known_train[final_cols]
    X_test_final = df_known_test[final_cols]
    rf = RandomForestClassifier(random_state=random_state)
    rf.fit(X_train_final, y_train)
    y_pred_final = rf.predict(X_test_final)
    test_error = sum(y_pred_final != y_test) / len(y_test)
    test_accuracy = sum(y_pred_final == y_test) / len(y_test)
    val_accuracy = 1 - best_val_error

    return selected_groups, best_val_error, test_error

# Main cross-validation routine
features_mut = generate_mutation_features("train_muts_data.csv", "train_features_v1.csv")
features_meth = generate_methylation_features("train_meth_data.csv", "train_features_v2.csv")

features_combined = features_mut.merge(features_meth,on='case_id',how='inner',suffixes=('_mut', '_meth'))
features_combined['Label'] = features_combined['Label_mut']
features_combined = features_combined.drop(columns=['Label_mut','Label_meth'])

k_splits = 5
results = []

for i in range(k_splits):
    df_train, df_val, df_test = split_data(features_combined, i)
    selected, val_error, test_error = greedy_feature_selection(df_train, df_val, df_test, random_state=i)
    results.append({'split': i + 1, 'features': selected, 'val_error': val_error, 'test_error': test_error, 'test_accuracy': 1 - test_error, 'val_accuracy': 1 - val_error})

for result in results:
    print(result)

# Create a mapping of feature names to their column lists
feature_location = {
    # All existing mutation groups
    "Total_Mutations": [col for col in df_train.columns if col == "total_mutations"],
    "Mutations_Per_Variant": [col for col in df_train.columns if col.startswith("var_count")],
    "Combi_Gene_Var": [col for col in df_train.columns if col.startswith("combi")],
    "STD_Gene": [col for col in df_train.columns if col == "std_mutations_per_gene"],
    "VAR_Gene": [col for col in df_train.columns if col == "var_mutations_per_gene"],
    "STD_Variant": [col for col in df_train.columns if col == "std_mutations_per_variant"],
    "VAR_Variant": [col for col in df_train.columns if col == "var_mutations_per_variant"],
    "Mutations_Per_Gene": [col for col in df_train.columns if col.startswith("mut_count_gene")],
    "Mutations_Per_Chromosome": [col for col in df_train.columns if col.startswith("mut_count_chr")],
    "Mutated_Genes": [col for col in df_train.columns if col == "num_mutated_genes"],
    "Driver_Genes": [col for col in df_train.columns if col == "prop_driver_mutations"],
    "TP53_Binary": [col for col in df_train.columns if col == "TP53_high_impact"],
    "MUC16_Binary": [col for col in df_train.columns if col == "MUC16_high_impact"],
    "MUC4_Binary": [col for col in df_train.columns if col == "MUC4_high_impact"],
    "AK2_MultiType": [col for col in df_train.columns if col == "AK2_multiple_variant_types"],
    "KMT2C_MultiType": [col for col in df_train.columns if col == "KMT2C_multiple_variant_types"],
    "Entropy": [col for col in df_train.columns if col == "gene_mutation_entropy"],
    "Combi_With_Mult": [col for col in df_train.columns if col == "gene_var_multiple_mutations"],
    "Del_To_Ins+Del": [col for col in df_train.columns if col == "del_of_delins_percentage"],
    "Reason": [col for col in df_train.columns if col == "reason_ratio"],
    "CA_Transversion": [col for col in df_train.columns if col == "CA_transversion_ratio"],
    # Additional methylation groups
    "Driver_Meth":        [col for col in df_train.columns if col == "prop_driver_methylations"],
    "Meth_Mean_Per_Gene":  [col for col in df_train.columns if col.startswith("meth_mean_gene_")],
    "Avg_CpG_Count":      [col for col in df_train.columns if col == "avg_CpG_count"],
    "Meth_STD_Gene":      [col for col in df_train.columns if col.startswith("meth_std_gene_")],
    "Meth_Probe_Count":   [col for col in df_train.columns if col.startswith("meth_probe_count_gene_")],
    "Meth_Entropy":       [col for col in df_train.columns if col == "gene_meth_entropy"],
    "Hyper_Meth":         [col for col in df_train.columns if col == "prop_hyper_methylations"],
    "Hypo_Meth":          [col for col in df_train.columns if col == "prop_hypo_methylations"]
}

feature_groups = [g for g, cols in feature_location.items() if len(cols) > 0]

# Choose the best model (split with lowest test error)
filtered_results = [x for x in results if x['test_error'] <= x['val_error']]
min_error = 1.0
best_result = None
for res in results:
    if res['test_error'] < min_error:
        best_result = res
        min_error = res['test_error']

# Extract the feature groups chosen by the best model
selected_groups = best_result['features']
final_cols = []
for group in selected_groups:
    final_cols.extend(feature_location[group])

# Generate features for the test set
test_features_mut = generate_mutation_features("test_muts_data.csv", "test_features_v1.csv")
test_features_meth = generate_methylation_features("test_meth_data.csv", "test_features_v2.csv")

# combine the test features
test_features_combined = test_features_mut.merge(test_features_meth,on='case_id',how='inner',suffixes=('_mut', '_meth'))

# Identify which columns are missing and create them all at once with zeros
missing_cols = [col for col in final_cols if col not in test_features_combined.columns]
if missing_cols:
    zeros_df = pd.DataFrame(0, index=test_features_combined.index, columns=missing_cols)
    test_features_combined = pd.concat([test_features_combined, zeros_df], axis=1)

# Prepare final training and test matrices
X_train_final = features_combined[final_cols]
y_train_final = features_combined['Label']
X_test_final = test_features_combined[final_cols]

# Fit the final Random Forest on all training data and predict on test data
rf_final = RandomForestClassifier(random_state=500)
rf_final.fit(X_train_final, y_train_final)
y_pred_test = rf_final.predict(X_test_final)

# Add predictions to the test features DataFrame and save to CSV
pred_df = pd.DataFrame({'id_case': test_features_combined['case_id'],'label_predict': y_pred_test})
pred_df.to_csv("meth_and_mut_preds.csv", index=False)
print("Saved predictions to meth_and_mut_preds.csv")
