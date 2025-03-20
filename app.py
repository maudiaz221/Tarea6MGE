import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Set Streamlit app title
st.title("Linear Regression Analysis")

# Load dataset directly from the given path
file_path = "data_athena/data.csv"
try:
    df = pd.read_csv(file_path)
    st.success("Dataset Loaded Successfully!")

    # Display dataset preview
    st.write("### Preview of Dataset")
    st.write(df.head())

    # Define regression function
    def run_regression(x, y, df):
        X = df[x]
        Y = df[y]
        X = sm.add_constant(X)  # Add intercept
        model = sm.OLS(Y, X).fit()

        # Display regression summary
        st.write(f"### Regression: {y} ~ {x}")
        st.text(model.summary())

        # Plot regression results
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.regplot(x=df[x], y=df[y], ax=ax, line_kws={'color': 'red'}, scatter_kws={'alpha': 0.5})
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{y} vs {x}")
        st.pyplot(fig)

    # Perform the three regressions
    run_regression('tasa_de_interes', 'tipo_de_cambio', df)
    run_regression('inflacion', 'tasa_de_interes', df)
    run_regression('inflacion', 'tipo_de_cambio', df)

except FileNotFoundError:
    st.error(f"Error: File not found at '{file_path}'. Please check the path.")