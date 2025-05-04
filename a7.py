import streamlit as st
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency

def load_data(file):
    if file is not None:
        return pd.read_csv(file)
    return None

def ks_drift(ref_df, curr_df, threshold=0.05):
    results = []
    common_cols = set(ref_df.columns) & set(curr_df.columns)
    num_cols = [col for col in common_cols if pd.api.types.is_numeric_dtype(ref_df[col])]

    for col in num_cols:
        stat, p_value = ks_2samp(ref_df[col].dropna(), curr_df[col].dropna())
        drift = p_value < threshold
        results.append({
            'Feature': col,
            'Type': 'Numerical',
            'Test': 'KS Test',
            'p-value': round(p_value, 4),
            'Drift Detected': drift
        })
    return results

def chi2_drift(ref_df, curr_df, threshold=0.05):
    results = []
    common_cols = set(ref_df.columns) & set(curr_df.columns)
    cat_cols = [col for col in common_cols if pd.api.types.is_object_dtype(ref_df[col]) or pd.api.types.is_categorical_dtype(ref_df[col])]

    for col in cat_cols:
        ref_counts = ref_df[col].value_counts()
        curr_counts = curr_df[col].value_counts()
        combined = pd.concat([ref_counts, curr_counts], axis=1).fillna(0)
        if combined.shape[0] < 2:
            continue # Skip if not enough categories
        chi2, p_value, _, _ = chi2_contingency(combined)
        drift = p_value < threshold
        results.append({
            'Feature': col,
            'Type': 'Categorical',
            'Test': 'Chi-square',
            'p-value': round(p_value, 4),
            'Drift Detected': drift
        })
    return results

def main():
    st.set_page_config(page_title="Full Drift Detector", layout="wide")
    st.title("Data Drift Detection (KS + Chi-Square Tests)")

    st.sidebar.header("Upload CSV Files")
    ref_file = st.sidebar.file_uploader("Upload Reference Dataset", type=["csv"])
    curr_file = st.sidebar.file_uploader("Upload Current Dataset", type=["csv"])

    if ref_file and curr_file:
        ref_df = load_data(ref_file)
        curr_df = load_data(curr_file)

        st.subheader("Reference Data Sample")
        st.dataframe(ref_df.head())

        st.subheader("Current Data Sample")
        st.dataframe(curr_df.head())

        with st.spinner("Running drift detection..."):
            ks_results = ks_drift(ref_df, curr_df)
            chi2_results = chi2_drift(ref_df, curr_df)
            all_results = ks_results + chi2_results
            results_df = pd.DataFrame(all_results)

        st.markdown("### Drift Detection Results")
        if not results_df.empty:
            st.dataframe(results_df)
            drifted = results_df[results_df['Drift Detected']]
            if not drifted.empty:
                st.error(f"{len(drifted)} feature(s) show significant drift.")
                st.dataframe(drifted)
            else:
                st.success("No significant drift detected.")
        else:
            st.warning("No common numerical or categorical columns to test.")
    else:
        st.info("Please upload both reference and current datasets.")

if __name__ == "__main__":
    main()
