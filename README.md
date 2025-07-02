# Deeper Roots Before the Storm

**Utilizing Machine Learning to Alert School Districts of Permanent School Closures**

Michael L. Chrzan, Francis A. Pearman, Benjamin W. Domingue
Stanford Graduate School of Education

## 📖 Overview

This repository accompanies the research paper *Deeper Roots Before the Storm*, which presents an early-warning system to help U.S. school districts anticipate the risk of mass school closures—defined as a district closing 10% or more of its schools—five years in advance. The project uses supervised machine learning models trained on administrative data from the National Center for Education Statistics (NCES) from 2000–2018.

The goal is to equip district and state leaders with proactive tools to support equitable, data-driven planning before closure decisions are made.

---

## 📊 Key Contributions

* Developed a **district-level predictive model** to forecast school closures five years in advance.
* Evaluated five machine learning methods:

  * Elastic Net Logistic Regression
  * Random Forest
  * XGBoost
  * LSTM Neural Networks
  * SuperLearner Ensemble
* Found **XGBoost** to be the most performant and generalizable, with passable AUC-PR and high recall.
* Conducted **error analysis** across subgroups (e.g., district type, urbanicity) to assess bias and improve equity.
* Provided **policy guidance** on using predictive models to enable equitable school closure planning.

---

## 🔍 Data

* Source: [NCES Common Core of Data (CCD)](https://nces.ed.gov/ccd/)
* Time range: 2000–2018
* Unit of analysis: District-year
* Features include:

  * Enrollment trends
  * School characteristics (type, diversity, Title I status)
  * Financial metrics (per-pupil expenditure, student-teacher ratio)
  * Demographics (race/ethnicity, FRL%)
  * Segregation indices (Theil Index)

Note: Due to the size of these educational data, raw datasets are **not included** in this repo. Data may be shared upon request.

---

## 🧠 Methods

* **Preprocessing**: Winsorization, handling missing data, aggregation to district level
* **Modeling**: Grouped cross-validation, class-weighted loss functions
* **Evaluation**:

  * Primary: AUC-PR (area under precision-recall curve)
  * Secondary: Recall (sensitivity), error analysis across district types/locales
* **Ethical Safeguards**:

  * Focus on district-level predictions to avoid reinforcing school-level bias
  * Careful use of demographic features to improve equity, not perpetuate harm

---

## 📈 Results Summary

| Model         | AUC-PR | Recall    |
| ------------- | ------ | --------- |
| **XGBoost**   | 0.396  | 0.678     |
| SuperLearner  | 0.393  | 0.454     |
| LSTM          | 0.391  | **0.776** |
| Random Forest | 0.374  | 0.684     |
| Elastic Net   | 0.369  | 0.724     |

* **XGBoost** offered the best tradeoff between predictive power and stability.
* **False positives** were more prevalent in small and charter districts.
* **False negatives** were more common in urban districts, a key equity concern.

---

## 🧭 Intended Use

This model is intended as a **planning tool** for districts and policymakers—not a prescriptive tool for deciding which schools to close. By providing early signals, it enables better community engagement, equity safeguards, and more thoughtful resource allocation.

---

## 🧑‍⚖️ Citation

If you use or reference this work, please cite:

> Chrzan, M. L., Pearman, F. A., & Domingue, B. W. (2025). *Deeper Roots Before the Storm: Utilizing Machine Learning to Alert School Districts of Permanent School Closures*. Stanford Graduate School of Education.

---

## 📬 Contact

For questions, collaborations, or data access:

* **Michael Chrzan** — \[[mlchrzan1@gmail.com](mailto:mlchrzan1@gmail.com)]
