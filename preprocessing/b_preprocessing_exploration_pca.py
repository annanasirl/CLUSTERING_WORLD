import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def run_pca(
        data_csv_path,
        top_n_loadings=5,
        output_txt_path="pca_results.txt"
):
    """
    Runs PCA on a standardized country x indicator dataset.
    """

    # Load data
    df = pd.read_csv(data_csv_path, index_col=0)
    X = df.values

    # Covariance matrix
    cov_matrix = np.cov(X, rowvar=False)

    # Eigenvalues & vectors
    eigvals, eigvecs = np.linalg.eig(cov_matrix)

    # Take only the real part
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    # Sort by descending eigenvalue
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Explained variance
    explained_var_ratio = eigvals / eigvals.sum()
    cum_explained_var = np.cumsum(explained_var_ratio)

    # Scree plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_var_ratio) + 1), explained_var_ratio, marker='o', label='Individual')
    plt.plot(range(1, len(explained_var_ratio) + 1), cum_explained_var, marker='s', label='Cumulative')
    plt.title('Scree Plot / Explained Variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.xticks(range(1, len(explained_var_ratio) + 1))
    plt.grid(True)
    plt.legend()
    plt.show()

    # Loadings
    loadings = pd.DataFrame(
        eigvecs,
        index=df.columns,
        columns=[f"PC{i + 1}" for i in range(len(eigvals))]
    )

    # === WRITE OUTPUT TO TXT FILE ===
    with open(output_txt_path, "w", encoding="utf-8") as f:

        f.write("Explained variance by component:\n")
        for i, var in enumerate(explained_var_ratio):
            f.write(
                f"PC{i + 1}: {var:.4f} "
                f"({cum_explained_var[i]:.4f} cumulative)\n"
            )

        f.write("\nTop contributing indicators per component:\n")

        for pc in loadings.columns:
            top = loadings[pc].abs().sort_values(ascending=False).head(top_n_loadings)

            f.write(f"\n{pc}:\n")
            for indicator, value in top.items():
                sign = "+" if loadings.at[indicator, pc] >= 0 else "-"
                f.write(f"  {indicator} ({sign}{abs(value):.3f})\n")

    return X, eigvals, eigvecs, loadings
