# USA Real Estate Data Cleaning and Preparation

## Project Overview

This project focuses on the critical first step of the data science lifecycle: **Data Quality Assurance and Cleaning**. Using the vast Kaggle USA Real Estate Dataset (2.2M+ records), I implemented a robust, multi-stage imputation strategy to transform noisy, incomplete, and non-standard raw data into a clean, analysis-ready feature set.

**Key Technical Skills Demonstrated:**
* Advanced Missing Value Imputation (Mode, Median, Group-wise Transformation)
* Non-Standard Data Handling (`pd.to_numeric(errors='coerce')`)
* Data Type Conversion and Error Handling (`IntCastingNaNError`)
* Feature Engineering (Creation of Missingness Indicator Flags)
* Scalable Data Handling with Pandas (2.2 Million Rows)

---

## 1. Initial Data Quality Assessment

The raw dataset contained over 2.2 million listings but suffered from significant sparsity, particularly in key numerical fields.

**Initial Strategy:** Instead of dropping rows and losing valuable geographic/price information, I opted for localized, data-informed imputation combined with flagging.

| Feature | Initial Missing % | Cleaning Strategy | Rationale |
| :--- | :--- | :--- | :--- |
| `bed`, `bath` | Moderate/High | Group-wise **Mode** imputation by `zip_code` (fallback to Global Mode) | Preserves integer nature; localized to immediate neighborhood norms. |
| `acre_lot` | High | Group-wise **Median** imputation by `zip_code` (fallback to Global Median) | Median is robust to extreme land-size outliers; localized to immediate area. |
| `house_size` | High | Group-wise **Median** imputation by `bed` and `bath` count. | House size is highly correlated with the number of rooms; this provides the most informed estimate. |

---

## 2. Imputation and Data Integrity

The core of the process focused on maximizing data integrity while ensuring the estimates were relevant to the property:

1.  **Non-Standard Value Coercion:** Used `pd.to_numeric(errors='coerce')` to convert non-standard missing markers (e.g., '-', ' ', 'NaN' strings) into proper `np.nan` values that the `fillna()` function could recognize.
2.  **Missingness Flagging:** Created indicator columns (e.g., `house_size_IMPUTED`) *before* imputation. This allows any future predictive model to learn from the fact that a value was missing, mitigating imputation bias.
3.  **Group-wise Transformation:** Leveraged `df.groupby(...).transform(lambda x: x.fillna(x.median()))` to ensure the imputed values were statistically relevant to their local context (e.g., median house size for 3-bed, 2-bath properties).
4.  **Final Cleanup:** Applied the global median/mode as a final fallback to eliminate all remaining stubborn `NaN` values that belonged to fully sparse groups.

---

## 3. Result and Validation

The cleaning process resulted in a feature-rich, fully numeric DataFrame ready for advanced regression modeling.

* **Zero Missing Values:** All target columns (`price`, `house_size`, `acre_lot`, `bed`, `bath`) now contain zero null values.
