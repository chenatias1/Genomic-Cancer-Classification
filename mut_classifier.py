import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Function to split data into train, validation, and test sets
def split_data(df_known, rand_i):
    # First, split into 80% (train+validation) and 20% (test)
    df_known_temp, df_known_test = train_test_split(
        df_known, test_size=0.2, stratify=df_known['Label'], random_state=rand_i)
    # Then, split the 80% into 50% train and 50% validation (i.e., 40%/40% of the original)
    df_known_train, df_known_validation = train_test_split(
        df_known_temp, test_size=0.5, stratify=df_known_temp['Label'], random_state=rand_i)
    return df_known_train, df_known_validation, df_known_test

def generate_mutation_features(file_path: str, genes_file: str = "100_genes.csv"):
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

    #4:  Create a binary feature indicating presence of a high-impact TP53 mutation
    tp53_high_impact = filtered_df[
        (filtered_df["Gene_name"] == "TP53") &
        (filtered_df["Variant_Classification"].isin(["Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins"]))
    ].groupby("case_id").size()
    tp53_high_impact = tp53_high_impact.apply(lambda x: 1).reindex(filtered_df["case_id"].unique(), fill_value=0).rename("TP53_high_impact")

    #5: Create a binary feature indicating presence of a high-impact MUC16 mutation
    muc16_high_impact = filtered_df[
        (filtered_df["Gene_name"] == "MUC16") &
        (filtered_df["Variant_Classification"].isin(["Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins"]))
    ].groupby("case_id").size()
    muc16_high_impact = muc16_high_impact.apply(lambda x: 1).reindex(filtered_df["case_id"].unique(), fill_value=0).rename("MUC16_high_impact")

    #6: Create a binary feature indicating presence of a high-impact MUC4 mutation
    muc4_high_impact = filtered_df[
        (filtered_df["Gene_name"] == "MUC4") &
        (filtered_df["Variant_Classification"].isin(["Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins"]))
    ].groupby("case_id").size()
    muc4_high_impact = muc4_high_impact.apply(lambda x: 1).reindex(filtered_df["case_id"].unique(), fill_value=0).rename("MUC4_high_impact")

    #7: For AK2 and KMT2C: count distinct Variant_Classification per patient, then binary if >1
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

    #16: G>T Transversion Ratio
    def is_GT_transversion(row):
        return int(row['Reference_Allele'] == 'G' and row['Tumor_Seq_Allele2'] == 'T')
    filtered_df['GT_transversion'] = filtered_df.apply(is_GT_transversion, axis=1)
    gt_counts = filtered_df.groupby('case_id')['GT_transversion'].sum()
    total_counts = filtered_df.groupby('case_id').size()
    gt_ratio = (gt_counts / total_counts).fillna(0).rename("GT_transversion_ratio")

    #17: A>T Transversion Ratio
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

    return features

def greedy_feature_selection(df_known_train, df_known_validation, df_known_test, tolerance=0.05, random_state=42):
    feature_location = {
        "Total_Mutations": [col for col in df_known_train.columns if col == "total_mutations"],
        "Mutations_Per_Variant": [col for col in df_known_train.columns if col.startswith("var_count_")],
        "Combi_Gene_Var": [col for col in df_known_train.columns if col.startswith("combi_")],
        "STD_Gene": [col for col in df_known_train.columns if col == "std_mutations_per_gene"],
        "VAR_Gene": [col for col in df_known_train.columns if col == "var_mutations_per_gene"],
        "STD_Variant": [col for col in df_known_train.columns if col == "std_mutations_per_variant"],
        "VAR_Variant": [col for col in df_known_train.columns if col == "var_mutations_per_variant"],
        "Mutations_Per_Gene": [col for col in df_known_train.columns if col.startswith("mut_count_gene_")],
        "Mutations_Per_Chromosome": [col for col in df_known_train.columns if col.startswith("mut_count_chr_")],
        "Mutated_Genes": [col for col in df_known_train.columns if col == "num_mutated_genes"],
        "Driver_Genes": [col for col in df_known_train.columns if col == "prop_driver_mutations"],
        "TP53_Binary": [col for col in df_known_train.columns if col == "TP53_high_impact"],
        "MUC16_Binary": [col for col in df_known_train.columns if col == "MUC16_high_impact"],
        "MUC4_Binary": [col for col in df_known_train.columns if col == "MUC4_high_impact"],
        "AK2_MultiType": [col for col in df_known_train.columns if col == "AK2_multiple_variant_types"],
        "KMT2C_MultiType": [col for col in df_known_train.columns if col == "KMT2C_multiple_variant_types"],
        "Entropy": [col for col in df_known_train.columns if col == "gene_mutation_entropy"],
        "Del_To_Ins+Del": [col for col in df_known_train.columns if col == "del_of_delins_percentage"],
        "Reason": [col for col in df_known_train.columns if col == "reason_ratio"],
        "CA_Transversion": [col for col in df_known_train.columns if col == "CA_transversion_ratio"],
        "GT_Transversion": [col for col in df_known_train.columns if col == "GT_transversion_ratio"],
        "AT_Transversion": [col for col in df_known_train.columns if col == "AT_transversion_ratio"]
    }

    feature_groups = list(feature_location.keys())
    selected_groups = []
    best_val_error = 1.0
    improved = True

    y_train = df_known_train['Label']
    y_val = df_known_validation['Label']

    # Greedy feature selection loop
    while improved:
        improved = False
        best_group = None
        for group in feature_groups:
            if group in selected_groups:
                continue
            current_cols = sum((feature_location[g] for g in selected_groups + [group]), [])
            rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
            rf.fit(df_known_train[current_cols], y_train)
            val_error = 1 - accuracy_score(y_val, rf.predict(df_known_validation[current_cols]))
            if val_error < best_val_error:
                best_val_error = val_error
                best_group = group
                improved = True
        if improved:
            selected_groups.append(best_group)

    # Train final model on the selected features
    final_cols = sum((feature_location[g] for g in selected_groups), [])
    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    rf.fit(df_known_train[final_cols], y_train)
    test_error = 1 - accuracy_score(df_known_test['Label'], rf.predict(df_known_test[final_cols]))
    return selected_groups, best_val_error, test_error

# Main cross-validation routine
features = generate_mutation_features("train_muts_data.csv")
k_splits = 5
results = []

for i in range(k_splits):
    df_train, df_val, df_test = split_data(features, i)
    selected, val_error, test_error = greedy_feature_selection(
        df_train, df_val, df_test, random_state=i
    )
    results.append({
        'split': i + 1,
        'features': selected,
        'val_error': val_error,
        'test_error': test_error,
        'test_accuracy': 1 - test_error,
        'val_accuracy': 1 - val_error
    })

for result in results:
    print(result)

# Create a mapping of feature names to their column lists
feature_location = {
    "Total_Mutations": [col for col in df_train.columns if col == "total_mutations"],
    "Mutations_Per_Variant": [col for col in df_train.columns if col.startswith("var_count_")],
    "Combi_Gene_Var": [col for col in df_train.columns if col.startswith("combi_")],
    "STD_Gene": [col for col in df_train.columns if col == "std_mutations_per_gene"],
    "VAR_Gene": [col for col in df_train.columns if col == "var_mutations_per_gene"],
    "STD_Variant": [col for col in df_train.columns if col == "std_mutations_per_variant"],
    "VAR_Variant": [col for col in df_train.columns if col == "var_mutations_per_variant"],
    "Mutations_Per_Gene": [col for col in df_train.columns if col.startswith("mut_count_gene_")],
    "Mutations_Per_Chromosome": [col for col in df_train.columns if col.startswith("mut_count_chr_")],
    "Mutated_Genes": [col for col in df_train.columns if col == "num_mutated_genes"],
    "Driver_Genes": [col for col in df_train.columns if col == "prop_driver_mutations"],
    "TP53_Binary": [col for col in df_train.columns if col == "TP53_high_impact"],
    "MUC16_Binary": [col for col in df_train.columns if col == "MUC16_high_impact"],
    "MUC4_Binary": [col for col in df_train.columns if col == "MUC4_high_impact"],
    "AK2_MultiType": [col for col in df_train.columns if col == "AK2_multiple_variant_types"],
    "KMT2C_MultiType": [col for col in df_train.columns if col == "KMT2C_multiple_variant_types"],
    "Entropy": [col for col in df_train.columns if col == "gene_mutation_entropy"],
    "Del_To_Ins+Del": [col for col in df_train.columns if col == "del_of_delins_percentage"],
    "Reason": [col for col in df_train.columns if col == "reason_ratio"],
    "CA_Transversion": [col for col in df_train.columns if col == "CA_transversion_ratio"],
    "GT_Transversion": [col for col in df_train.columns if col == "GT_transversion_ratio"],
    "AT_Transversion": [col for col in df_train.columns if col == "AT_transversion_ratio"]
}

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
test_features = generate_mutation_features("test_muts_data.csv")

# Identify which columns are missing and create them all at once with zeros
missing_cols = [col for col in final_cols if col not in test_features.columns]
if missing_cols:
    zeros_df = pd.DataFrame(0, index=test_features.index, columns=missing_cols)
    test_features = pd.concat([test_features, zeros_df], axis=1)

# Prepare final training and test matrices
X_train_final = features[final_cols]
y_train_final = features['Label']
X_test_final = test_features[final_cols]

# Fit the final Random Forest on all training data and predict on test data
rf_final = RandomForestClassifier(random_state=500, n_jobs=-1)
rf_final.fit(X_train_final, y_train_final)
y_pred_test = rf_final.predict(X_test_final)

# Add predictions to the test features DataFrame and save to CSV
test_features['predict_label'] = y_pred_test
test_features[['case_id','predict_label']].to_csv("mut_preds.csv", index=False)

## plot C

def plot_variant_hist(df1, df2):
    # Define the category groups and their colors
    group_definitions = {
        "Non-Silent": {
            "categories": [
                "Missense_Mutation", "In_Frame_Del", "Nonsense_Mutation", "Frame_Shift_Del",
                "Frame_Shift_Ins", "In_Frame_Ins", "Translation_Start_Site", "Nonstop_Mutation"
            ],
            "color": "blue"
        },
        "UTR": {
            "categories": ["3'UTR", "5'UTR"],
            "color": "red"
        },
        "Intron": {
            "categories": ["Intron"],
            "color": "green"
        },
        "Silent": {
            "categories": ["Silent"],
            "color": "purple"
        },
        "Flank": {
            "categories": ["3'Flank", "5'Flank"],
            "color": "orange"
        },
        "Splice": {
            "categories": ["Splice_Site", "Splice_Region"],
            "color": "brown"
        }
    }

    # Flatten all categories and create mapping from category to group and color
    category_to_group = {}
    category_to_color = {}
    for group, info in group_definitions.items(): # group is key, info is the category list or color
        for cat in info["categories"]: # run on ech of the varients in the group
            category_to_group[cat] = group
            category_to_color[cat] = info["color"] # for each varient of each category assign the correct color

    # Build the list of all varients
    all_categories = list(category_to_group.keys())
    col_names = [f"var_count_{cat}" for cat in all_categories]

    # Concatenate both files
    df = pd.concat([df1, df2], ignore_index=True)

    # Only keep relevant columns that exist
    available_cols = [col for col in col_names if col in df.columns]

    # a dictionary containing the sum of mutations of each varient
    sums = {col.replace("var_count_", ""): df[col].sum() for col in available_cols}

    # Organize categories by group and sort within each group
    grouped_sorted = []
    for group, info in group_definitions.items():
        group_cats = [cat for cat in info["categories"] if cat in sums]
        # Sort by sum descending
        sorted_cats = sorted(group_cats, key=lambda x: sums.get(x, 0), reverse=True)
        for cat in sorted_cats:
            grouped_sorted.append(cat)

    # Prepare plotting data
    labels = grouped_sorted # so labels will appear at the same order as the sorted varients
    values = [sums.get(cat, 0) for cat in labels] # saving each amount of muts from dictionary to vector
    colors = [category_to_color[cat] for cat in labels]
    group_labels = [category_to_group[cat] for cat in labels] #for legend later

    # Plot
    plt.figure(figsize=(14, 7))
    bars = plt.bar(labels, values, color=colors)

    # Create custom legend
    from matplotlib.patches import Patch
    legend_patches = []
    for group, info in group_definitions.items():
        legend_patches.append(Patch(color=info["color"], label=group))
    plt.legend(handles=legend_patches, title="Category Group")

    plt.title("Total Mutation Counts per Varient Type")
    plt.ylabel("Number of Mutations")
    plt.xlabel("Variant Classification")
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()
    plt.show()

def align_train_test_features(train,test):

    all_features = sorted(set(train.columns).union(set(test.columns)) - {"case_id"})
    train_aligned = train.set_index("case_id").reindex(columns=all_features, fill_value=0).reset_index()
    test_aligned = test.set_index("case_id").reindex(columns=all_features, fill_value=0).reset_index()

    return train_aligned, test_aligned

