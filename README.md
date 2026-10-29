# Room Occupancy Forecasting with Decision Tree Regression

**MMA3001 Numerical Methods and Machine Learning, final project. Author: [Your name and student ID]**

This project predicts how many people will be in a room **15 minutes ahead** from historical occupancy measurements and time-based variables, so that a heating, ventilation and air-conditioning (HVAC) system can respond before occupancy changes. The main model is a **Decision Tree Regression**, compared against a Historical Average, Linear Regression and Support Vector Regression (SVR). The data are Monash Smart Infrastructure occupancy sensor readings (about 9 million rows, five rooms); the models are trained and tested on **one room**.

> ## ⚠️ The data file is NOT in this repository
>
> The occupancy data is too large for GitHub (the zip is about 85 MB and unzips to a **1.6 GB** CSV), so you must **download it yourself** before running any code.
>
> 1. Download **`5occupancySensor_MayToDec2024_9MRows.zip`** from [LINK TO WHERE THE FILE IS HOSTED, e.g. Moodle / OneDrive / Google Drive].
> 2. Unzip it. It contains one file: `5occupancySensor_MayToDec2024_9MRows.csv`.
> 3. Place the CSV at **`data/raw/5occupancySensor_MayToDec2024_9MRows.csv`** (create the folder if needed).
>
> The data belong to Monash University and were supplied by the teaching team for this unit. Do not redistribute them or commit them to Git.

**In brief.** The decision tree had the lowest RMSE (0.578) and highest R² (0.452) on the unseen test set, and an MAE (0.286) close to the best, which was SVR's (0.268). It is also fast (0.040 s runtime) and easy to interpret. The R² shows that a good share of the variation in occupancy is still unexplained, so more work would be needed before using it in a real building system.

**Contents:** [1. Engineering problem](#1-engineering-problem) | [2. Input and output data](#2-input-and-output-data) | [3. Computational solution](#3-computational-solution) | [4. Alternative solutions](#4-alternative-solutions) | [5. Validation and results](#5-validation-and-results) | [6. Optimisation and performance](#6-optimisation-and-performance) | [7. Limitations](#7-limitations) | [8. Getting started](#8-getting-started) | [9. Repository layout](#9-repository-layout) | [10. AI use](#10-ai-use) | [11. Licence and data](#11-licence-and-data)

## 1. Engineering problem

HVAC systems keep a building comfortable, but the number of people in a room changes a lot through the day. Systems that run on fixed schedules can react slowly to new occupancy or condition empty rooms, which wastes energy. A numerical model that predicts occupancy a short time ahead can make the system more responsive.

**Aim.** Develop a decision tree regression model that predicts room occupancy 15 minutes in advance from historical occupancy and time-based variables, and evaluate it against Historical Average, Linear Regression and SVR.

**Scope.** The dataset covers 5 rooms, but the models are trained and validated on **one room only**. It is assumed (not tested) that the approach could be retrained on the other rooms with similar success. Limitations include missing sensor measurements and the model's difficulty with unexpected changes in room use.

## 2. Input and output data

### Input data

| Variable | Meaning | Unit | Allowed values |
|---|---|---|---|
| `floorspaceid` | Which room | Text | Any room identifier |
| `headcount` | Total occupancy | Integer | Zero or more |
| `collecteddate` | Time of the reading | Date and time | Any date and time |

Only these three columns are used. `floorspaceid` is used early to filter the data down to the selected room, `headcount` becomes the room occupancy, and `collecteddate` is split into date, hour and minute.

### Output data

| Output | Meaning | Unit | Intended use |
|---|---|---|---|
| Input time | Current time | Any 15-minute interval | See the occupancy at that time |
| Current occupancy | Occupancy at that time | Positive integer | Lets the model be compared with the prediction |
| Predicted occupancy | The model's prediction 15 minutes after the input time | Positive integer | Value passed to the HVAC system |

### Data processing

- Timestamps are split up and sorted into **15-minute intervals**.
- Invalid observations (for example rows with a missing head count) and duplicate timestamps are removed.
- When there is no new occupancy update, the **previous value is kept for up to 30 minutes**, after which the room is **assumed to be unoccupied**. This keeps the dataset continuous without immediately treating missing measurements as zero. If occupancy actually changed during that period, the retained value is wrong, which adds uncertainty to the inputs and may reduce accuracy.

## 3. Computational solution

**Method.** Decision Tree Regression is a supervised learning technique for predicting continuous values. It repeatedly splits the training data into smaller groups using threshold rules on the input variables, choosing splits that reduce variation in the target. The prediction is the mean occupancy of the observations in the final group (leaf). It suits this problem because occupancy has non-linear relationships with time and recent occupancy, and changes in steps, much like a tree.

**Implementation.** Python, with `pandas` for processing and scikit-learn's `DecisionTreeRegressor`.

**Features.** Six features are built from occupancy and time:

1. Current occupancy
2. Occupancy 15 minutes earlier
3. Occupancy 30 minutes earlier
4. Minute
5. Hour
6. Day

**Data split.** 70% training, 15% validation, 15% testing. The training set builds the decision rules, the validation set is used to choose tree depth and leaf size, and the test set is used once to compare models on unseen data.

**Overfitting.** A tree that is too deep or has too many leaves can fit noise. A simple, accurate model is preferred.

**Assumptions.** Historical occupancy and time patterns are enough to predict occupancy 15 minutes ahead, and a room with no new reading for 30 minutes is empty.

## 4. Alternative solutions

Three alternatives were tested to check that a decision tree was the best choice. They were compared on accuracy, computational complexity, suitability for occupancy patterns and implementation effort.

| Method | How it was used | Strengths | Weaknesses |
|---|---|---|---|
| **Historical Average** | Mean occupancy for that 15-minute time and day of the week (the baseline) | Very simple, low memory, scales easily | Cannot react to unexpected changes |
| **Linear Regression** | Uses the six numerical features | Fast, low memory | Assumes an additive linear relationship, so it struggles with non-linear patterns |
| **Support Vector Regression (SVR)** | RBF kernel, inputs standardised before training | Can approximate smooth non-linear relationships | Needs more computation, harder to interpret, parameters must be re-tuned |

## 5. Validation and results

Models were scored on the unseen **test set** using three metrics:

- **MAE** (mean absolute error): average absolute difference between predicted and actual occupancy.
- **RMSE** (root mean squared error): penalises large errors more, which matters for HVAC control.
- **R²** (coefficient of determination): how much of the variation in occupancy the model explains compared with always predicting the mean.

| Model | Test MAE | Test RMSE | Test R² |
|---|---|---|---|
| **Decision Tree Regression** | 0.286 | **0.578** | **0.452** |
| Historical Average | 0.448 | 0.705 | 0.185 |
| Linear Regression | 0.307 | 0.590 | 0.429 |
| Support Vector Regression | **0.268** | 0.590 | 0.429 |

- SVR has the lowest MAE (0.268), with the decision tree second (0.286). The gap of 0.018 people is small.
- The decision tree has the lowest RMSE, so it makes fewer large errors, which matters because large errors can cause inappropriate HVAC responses.
- The decision tree has the highest R² (0.452). Historical Average and Linear Regression are worse than the tree and SVR, consistent with them struggling with non-linear patterns and unexpected changes.
- The decision tree was chosen because it has the lowest RMSE, the highest R² and an MAE close to SVR's, and its threshold rules make it easier to see how past occupancy and time affect predictions.
- An R² of 0.452 means much of the variation is still unexplained. Possible causes are unexpected changes in room use, limited input variables and errors introduced by the way missing data was handled.

## 6. Optimisation and performance

**Tree depth.** Depths of 4, 6, 8, 10, 12, 15 and 20 were tested on the validation set, and the one with the lowest MAE was chosen: **20**. Choosing the largest depth tested may mean the data has little noise, or that deeper values should also have been tried.

**Other settings.**

| Setting | Value | Reason |
|---|---|---|
| `min_samples_leaf` | 250 (fixed) | Limits the influence of outliers on the rules |
| `random_state` | 42 | Makes results reproducible, including for other rooms |

**Runtime and memory.**

| Model | Total runtime (s) | Peak allocations (MiB) |
|---|---|---|
| Decision Tree | 0.040 | 4.08 |
| SVR | 40.474 | 2.16 |
| Historical Average | 0.084 | 6.52 |
| Linear Regression | 0.011 | 9.06 |

The decision tree is fast and uses little memory, so computational cost is not a concern. SVR is far slower (about 40 s).

**Ways to improve the model.**

1. Find a better way to fill gaps in the occupancy data than assuming zero, which can mislead the model.
2. Train on more rooms to increase the sample size and make the model usable in multiple rooms and buildings.
3. Test several values of `min_samples_leaf` instead of fixing it at 250, which may reduce MAE further.

## 7. Limitations

- **One room.** Results cannot be assumed to hold for the other rooms or for other buildings.
- **Missing data.** Gaps are filled by carrying the last value forward for 30 minutes and then assuming zero. Reported errors may therefore not fully reflect performance on real measurements.
- **Unexpected changes.** The model struggles to predict sudden changes in how a room is used, and decision trees can overfit.
- **Moderate R².** At 0.452, further improvement would be needed before using the model in an operational building-management system.

## 8. Getting started

**Requirements.** Python 3 with `pandas`, `numpy`, `scikit-learn` and `jupyter` [edit to match your code; list them in `requirements.txt`].

```bash
git clone https://github.com/jsim0061/FInal_MMA3001.git
cd FInal_MMA3001

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Download the data (required).** Download `5occupancySensor_MayToDec2024_9MRows.zip`, unzip it, and put the CSV here:

```
data/raw/5occupancySensor_MayToDec2024_9MRows.csv
```

Without this file the code fails with a "file not found" error. The CSV is about 1.6 GB, so check you have disk space and allow a few minutes to load it.

**Run the analysis.**

```bash
# [Replace with your actual notebook or script names, for example:]
jupyter notebook                     # open the notebook and run all cells from top to bottom
```

## 9. Repository layout

[Edit to match your repository.]

```
data/raw/         put the downloaded CSV here (git-ignored)
[notebook or script]   data preparation, models and evaluation
requirements.txt  Python dependencies
README.md         this file
```

## 10. AI use

ChatGPT was used to help generate and structure Python code (data preprocessing, Decision Tree Regression, Linear Regression, Historical Average and SVR), explain machine learning concepts, investigate model parameters, resolve programming errors, suggest improvements to the written report, and develop this GitHub repository structure.

AI-generated code was checked by running it and confirming the output was correct, and the results were inspected using MAE, RMSE and R². Problems encountered included errors in the generated Python code, package installation issues, missing dataset columns, and inconsistencies between model parameters and documentation. The project scope, choice of methods, testing, parameter settings and the content of the report were my own decisions.

## 11. Licence and data

[State your code licence, e.g. MIT, and add a `LICENSE` file.] The data belong to Monash University, were supplied by the teaching team for this unit, and are not covered by that licence or included in this repository. Do not redistribute them.
