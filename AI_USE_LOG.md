# AI Use Log — MMA3001 Occupancy Prediction

**AI tool:** ChatGPT (OpenAI)  
**Project:** Predicting room occupancy 15 minutes ahead using Decision Tree Regression  
**Course:** MMA3001 — Numerical Methods and Machine Learning

## 1. Declaration

I used ChatGPT extensively to assist with planning, coding, debugging, testing, documentation and report drafting for this project. Its contribution was substantial, especially for initial Python implementations and supporting GitHub files. The final submission, the interpretation of results and the accuracy of claims remain my responsibility.

## 2. Record of AI assistance

| Project activity | How ChatGPT assisted | Approximate contribution |
|---|---|---|
| Project planning | Explained the assignment requirements and suggested a manageable workflow | Moderate |
| Data preprocessing | Drafted code for reading sensor data, selecting a room, creating 15-minute intervals and handling missing values | High |
| Decision Tree Regression | Generated model code, chronological splits, prediction outputs and depth investigation | High |
| Alternative models | Generated Linear Regression, Historical Average and Support Vector Regression implementations | High |
| Model evaluation | Explained MAE, RMSE, R², model comparisons and validation limitations | Moderate to high |
| Performance analysis | Suggested and implemented approaches for timing and memory profiling | High |
| Debugging | Helped diagnose missing files, missing columns, Python interpreter issues and package imports | Moderate to high |
| GitHub repository | Drafted README material, organised files, and assisted with tests, docstrings and HTML documentation | High |
| Written report | Suggested section outlines, example paragraphs and interpretations for the project report | Moderate to high |

## 3. Verification and critical evaluation

AI-produced code was checked by attempting to execute scripts, investigating error messages and reviewing model outputs. The regression methods were compared using MAE, RMSE and R². Automated tests were also added to check selected preprocessing and model behaviours.

Verification has limitations: passing unit tests does **not** by itself establish that the final models reproduce every result in the written report or that predictions represent independently verified physical occupancy. Before submission, the final scripts, model settings, numerical results and generated documentation should be checked for consistency.

## 4. Issues and limitations identified

- Some generated scripts initially failed because of incorrect paths, unavailable ZIP files or missing dataset columns.
- Different Python installations caused `ModuleNotFoundError` for Pandas in VS Code; Anaconda interpreter configuration had to be investigated.
- Versions of the Decision Tree script and the written report contained different `max_depth` and `min_samples_leaf` settings; the final version must be reconciled.
- The preprocessing rule assuming zero occupancy after more than 30 minutes without a reading is an assumption, not a confirmed sensor measurement, and may affect reported accuracy.
- The final model comparisons require checking that all methods use equivalent evaluation data. SVR may have used a smaller training sample for computational feasibility.

## 5. Decisions and responsibility

The investigation focuses on a 15-minute prediction horizon and uses Decision Tree Regression as the main method, with Historical Average, Linear Regression and SVR as comparisons. AI helped develop and explain these approaches, but the engineering purpose, use of the results, conclusions and final submission must be reviewed and defended by me.

## 6. Reflection

ChatGPT accelerated the creation of working Python scripts and helped me understand unfamiliar machine-learning and programming concepts. It was also useful for identifying approaches to documentation and testing. However, the debugging process and differences between script versions showed that AI-generated material can contain errors, inconsistent assumptions and incomplete verification. This project highlighted the need to understand the code, run independent checks and base engineering conclusions on reproducible results rather than trusting generated explanations alone.

---

**Before submitting:** Edit this log to reflect only verification and decisions you personally completed. Record any additional AI tools used, and ensure the report's Section 7.0 and this file are consistent.
