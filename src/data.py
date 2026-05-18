"""
Data loading and preprocessing for GSE139555 (Wu et al. 2020).

Pipeline:
    raw cellranger MTX (per sample)  -->  per-sample AnnData
    concatenate across samples       -->  combined AnnData
    join with TCR metadata           -->  obs columns (patient, tissue, clonotype, ... etc)
    build expansion label per patient (clonotype freq >= 2 = expanded)
    QC + normalize + select HVGs
    patient-stratified train/val/test split
"""

from __future__ import annotations

import os
import tarfile
import urllib.request
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE139nnn/GSE139555/suppl"
RAW_TAR_URL = f"{GEO_BASE}/GSE139555_RAW.tar"
TCELL_META_URL = f"{GEO_BASE}/GSE139555_tcell_metadata.txt.gz"


def download_file(url: str, dest: Path, force: bool = False) -> Path:
    # download a file to dest if it doesn't already exist
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"[skip] {dest} already exists ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    print(f"[download] {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"[done] {dest.stat().st_size / 1e6:.1f} MB")
    return dest


def extract_tar(tar_path: Path, out_dir: Path) -> Path:
    # extract a tar archive once
    tar_path = Path(tar_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sentinel = out_dir / ".extracted"
    if sentinel.exists():
        print(f"[skip] {out_dir} already extracted")
        return out_dir
    print(f"[extract] {tar_path} -> {out_dir}")
    with tarfile.open(tar_path) as tar:
        tar.extractall(out_dir)
    sentinel.touch()
    return out_dir


def list_sample_prefixes(raw_dir: Path) -> list[str]:
    """
    -GSE139555_RAW contains files like
        -GSM4143655_SAM24345862-lt1.barcodes.tsv.gz
        -GSM4143655_SAM24345862-lt1.matrix.mtx.gz
        -GSM4143655_SAM24345862-lt1.filtered_contig_annotations.csv.gz
    -Returns the list of distinct `GSMxxxxxxx_SAMxxxxxxxx-<sample>` prefixes.
    """
    raw_dir = Path(raw_dir)
    prefixes = set()
    for f in raw_dir.iterdir():
        name = f.name
        for suffix in (".barcodes.tsv.gz", ".features.tsv.gz", ".genes.tsv.gz", ".matrix.mtx.gz"):
            if name.endswith(suffix):
                prefixes.add(name[: -len(suffix)])
                break
    return sorted(prefixes)


def _sample_name_from_prefix(prefix: str) -> str:
    #extract the short sample id used in the metadata barcodes.
    tail = prefix.rsplit("-", 1)[-1]
    return tail.upper()


def load_sample(raw_dir: Path, prefix: str) -> ad.AnnData:
    # load one sample's MTX matrix into AnnData
    raw_dir = Path(raw_dir)
    barcodes_p = raw_dir / f"{prefix}.barcodes.tsv.gz"
    matrix_p = raw_dir / f"{prefix}.matrix.mtx.gz"
    features_p = raw_dir / f"{prefix}.features.tsv.gz"
    if not features_p.exists():
        features_p = raw_dir / f"{prefix}.genes.tsv.gz"

    barcodes = pd.read_csv(barcodes_p, header=None, sep="\t")[0].tolist()
    features = pd.read_csv(features_p, header=None, sep="\t")
    # cellranger features.tsv: gene_id, gene_symbol, type
    if features.shape[1] >= 2:
        var = pd.DataFrame({"gene_id": features[0].values}, index=features[1].astype(str).values)
    else:
        var = pd.DataFrame(index=features[0].astype(str).values)
    var.index.name = "gene_symbol"
    var_names_make_unique = pd.Index(var.index).duplicated().any()

    # scanpy/mmread handles .mtx.gz 
    X = sc.read_mtx(str(matrix_p)).X
    # cellranger MTX is (genes, cells) transpose to (cells, genes)
    X = X.T.tocsr()

    sample_name = _sample_name_from_prefix(prefix)
    obs = pd.DataFrame(
        {"sample": sample_name},
        index=[f"{sample_name}_{bc}" for bc in barcodes],
    )

    adata = ad.AnnData(X=X, obs=obs, var=var)
    if var_names_make_unique:
        adata.var_names_make_unique()
    return adata


def load_all_samples(raw_dir: Path, subset: list[str] | None = None) -> ad.AnnData:
    # load all (or a subset of) samples and concatenate into one AnnData
    prefixes = list_sample_prefixes(raw_dir)
    if subset is not None:
        subset_lc = {s.lower() for s in subset}
        prefixes = [p for p in prefixes if _sample_name_from_prefix(p).lower() in subset_lc]
    print(f"[load] {len(prefixes)} samples")
    adatas = []
    for p in prefixes:
        a = load_sample(raw_dir, p)
        print(f"  {p}: {a.n_obs} cells x {a.n_vars} genes")
        adatas.append(a)
    # join='inner' keeps only genes present in every sample
    combined = ad.concat(adatas, join="inner", merge="same", index_unique=None)
    return combined

def load_tcell_metadata(meta_path: Path) -> pd.DataFrame:
    #  load the T cell metadata table from GEO, index is the cell barcode
    df = pd.read_csv(meta_path, sep="\t", index_col=0)
    print(f"[meta] {len(df)} cells, columns: {list(df.columns)}")
    return df

def attach_metadata(adata: ad.AnnData, meta: pd.DataFrame) -> ad.AnnData:
    #keep only cells in the metadata table and copy its columns onto `adata.obs`
    common = adata.obs_names.intersection(meta.index)
    print(f"[merge] {len(common)} / {adata.n_obs} cells matched metadata "
          f"(metadata has {len(meta)} total)")
    if len(common) == 0:
        print("  adata sample of obs_names:", list(adata.obs_names[:5]))
        print("  meta sample of index:    ", list(meta.index[:5]))
        raise RuntimeError("No barcodes matched. Check the prefixing scheme.")
    adata = adata[common].copy()
    for col in meta.columns:
        adata.obs[col] = meta.loc[common, col].values
    return adata


def build_expansion_label(
    adata: ad.AnnData,
    clonotype_col: str,
    patient_col: str = "patient",
    min_expanded: int = 2,
) -> ad.AnnData:
    """
    Adds:
        -clone_size (per patient) and 
        -expanded (bool) columns to obs

    a clonotype is considered "expanded" if at least min_expanded cells from
    the *same patient* share it && cells without a clonotype call are dropped
    """
    if clonotype_col not in adata.obs.columns:
        raise KeyError(f"clonotype column {clonotype_col!r} not in obs: {list(adata.obs.columns)}")
    valid = adata.obs[clonotype_col].notna() & (adata.obs[clonotype_col].astype(str) != "")
    n_drop = (~valid).sum()
    if n_drop:
        print(f"[label] dropping {n_drop} cells without a clonotype")
    adata = adata[valid].copy()

    sizes = (
        adata.obs.groupby([patient_col, clonotype_col], observed=True)
        .size()
        .rename("clone_size")
    )
    adata.obs = adata.obs.join(sizes, on=[patient_col, clonotype_col])
    adata.obs["expanded"] = (adata.obs["clone_size"] >= min_expanded).astype(int)
    frac = adata.obs["expanded"].mean()
    print(f"[label] expanded fraction: {frac:.3f} ({adata.obs['expanded'].sum()} / {len(adata.obs)})")
    return adata


def qc_and_normalize(
    adata: ad.AnnData,
    min_genes: int = 200,
    min_cells: int = 3,
    max_pct_mito: float = 15.0,
    target_sum: float = 1e4,
    n_top_genes: int = 2000,
) -> ad.AnnData:
    """
    standard scanpy preprocessing workglow, returns an AnnData where:
        adata.X is log1p-normalized, restricted to highly variable genes
        adata.layers['counts'] keeps the raw counts (HVG-subsetted)
        adata.raw stores the pre-HVG log-normalized matrix (all genes)
    """
    print(f"[qc] start: {adata.n_obs} cells x {adata.n_vars} genes")

    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, log1p=False)

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    adata = adata[adata.obs["pct_counts_mt"] < max_pct_mito].copy()
    print(f"[qc] after filters: {adata.n_obs} cells x {adata.n_vars} genes")

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    adata.raw = adata

    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat", subset=True)
    print(f"[hvg] kept {adata.n_vars} highly variable genes")
    return adata


def patient_split(
    adata: ad.AnnData,
    patient_col: str = "patient",
    val_frac: float = 0.15,
    test_frac: float = 0.20,
    seed: int = 0,
) -> ad.AnnData:
    """
    Assign each patient (entire patient, not per-cell) to train/val/test

    adds adata.obs['split'] with values in {'train','val','test'}
    
    splitting by patient avoids clonotype leakage and reflects model if applied to a new patient
    """
    rng = np.random.default_rng(seed)
    patients = np.array(sorted(adata.obs[patient_col].unique()))
    rng.shuffle(patients)
    n = len(patients)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    test_pts = set(patients[:n_test])
    val_pts = set(patients[n_test : n_test + n_val])
    train_pts = set(patients[n_test + n_val :])

    def assign(p):
        if p in test_pts:
            return "test"
        if p in val_pts:
            return "val"
        return "train"

    adata.obs["split"] = adata.obs[patient_col].map(assign).astype("category")
    counts = adata.obs["split"].value_counts().to_dict()
    print(f"[split] train={len(train_pts)} val={len(val_pts)} test={len(test_pts)} patients")
    print(f"[split] cells per split: {counts}")
    return adata


def to_matrix(adata: ad.AnnData) -> np.ndarray:
    #return a dense float32 expression matrix (n_cells, n_genes)
    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)
