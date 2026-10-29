"""Shared model helpers: data validation, metrics, profiling and display."""
import pickle
import time
import tracemalloc
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = ['current_occupancy','previous_15min','previous_30min','hour','minute','day_of_week']
TARGET='next_15min_occupancy'

def load_data(path):
    """Load a cleaned 15-minute CSV with the six standard model features.

    Args:
        path (str or Path): Path to the processed occupancy CSV.
    Returns:
        pandas.DataFrame: Chronologically ordered observations, with UTC target times.
    Raises:
        ValueError: If required columns are missing or insufficient valid rows exist.
    """
    df=pd.read_csv(path)
    required=FEATURES+[TARGET,'collecteddate']
    missing=sorted(set(required)-set(df.columns))
    if missing:raise ValueError(f'Missing required columns: {missing}')
    df['collecteddate']=pd.to_datetime(df['collecteddate'],utc=True,errors='coerce')
    df['target_time']=pd.to_datetime(df['target_time'],utc=True,errors='coerce') if 'target_time' in df else df['collecteddate']+pd.Timedelta(minutes=15)
    df=df.dropna(subset=required+['target_time']).sort_values('collecteddate').reset_index(drop=True)
    if len(df)<20:raise ValueError('At least 20 valid observations are required')
    if 'target_is_measured' in df:
        df['target_is_measured']=df['target_is_measured'].astype(str).str.lower().eq('true')
    return df

def chronological_split(df):
    """Split observations 70/15/15 and drop labels crossing split boundaries.

    Args:
        df (pandas.DataFrame): Chronologically sorted data with target_time.
    Returns:
        tuple: Train, validation and test DataFrames.
    """
    n=len(df); a=int(n*.70); b=int(n*.85)
    train=df.iloc[:a].copy(); val=df.iloc[a:b].copy(); test=df.iloc[b:].copy()
    train=train.loc[train['target_time']<val['collecteddate'].iloc[0]].copy()
    val=val.loc[val['target_time']<test['collecteddate'].iloc[0]].copy()
    if min(len(train),len(val),len(test))<2:raise ValueError('Insufficient observations after boundary exclusion')
    return train,val,test

def scores(y_true,y_pred):
    """Return MAE, RMSE and R2 for continuous headcount predictions.

    Args:
        y_true (array-like): Sensor or processed target headcounts.
        y_pred (array-like): Predicted continuous headcounts.
    Returns:
        dict: MAE and RMSE in occupants; dimensionless R2.
    """
    return {'MAE':float(mean_absolute_error(y_true,y_pred)),
            'RMSE':float(np.sqrt(mean_squared_error(y_true,y_pred))),
            'R2':float(r2_score(y_true,y_pred))}


def section(title):
    """Print a readable section heading."""
    print("\n" + "=" * 76)
    print("  " + title)
    print("=" * 76)


def table(headers, rows, numeric_columns=()):
    """Print an aligned table; accepts preformatted strings or numeric scalars."""
    rows = [[str(value) for value in row] for row in rows]
    headers = [str(value) for value in headers]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
              for i in range(len(headers))]
    def format_row(row):
        return "  " + " | ".join(
            value.rjust(widths[i]) if i in numeric_columns else value.ljust(widths[i])
            for i, value in enumerate(row))
    print(format_row(headers))
    print("  " + "-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))


def scores_table(records):
    """Display MAE, RMSE (people), and R-squared at three decimal places."""
    section("REGRESSION ACCURACY  |  15-minute forecast")
    rows = []
    for item in records:
        split = item.get("Dataset", item.get("split", "Unknown"))
        count = item.get("n", item.get("N", "—"))
        count = f"{int(count):,}" if count != "—" else "—"
        rows.append([split, count, f"{float(item['MAE']):.3f}",
                     f"{float(item['RMSE']):.3f}", f"{float(item['R2']):.3f}"])
    table(["Dataset", "Records", "MAE (people)", "RMSE (people)", "R²"],
          rows, numeric_columns=(1, 2, 3, 4))
    print("  Lower MAE/RMSE is better; higher R² is better.")


def performance_table(row):
    """Display profiling values with consistent units and explicit limitations."""
    section(f"COMPUTATIONAL PERFORMANCE  |  {row['Model']}")
    metrics = [
        ("Model training/tuning", "Training_Time_s", "seconds", 3),
        ("Test-set prediction", "Test_Prediction_Time_s", "seconds", 3),
        ("Peak Python allocations (train)", "Training_Peak_Python_MiB", "MiB", 2),
        ("Peak Python allocations (test)", "Test_Prediction_Peak_Python_MiB", "MiB", 2),
        ("Estimated saved model size", "Model_Size_KiB", "KiB", 2),
    ]
    table(["Measure", "Value", "Unit"], [
        [label, f"{row[key]:.{precision}f}" if row.get(key) is not None else "N/A", unit]
        for label, key, unit, precision in metrics], numeric_columns=(1,))
    print("  Memory = tracemalloc allocations, NOT total system RAM.")
    print("  Times and memory depend on dataset, computer and training settings.")


def profile_call(func, *args, **kwargs):
    """Run func once, returning (result, seconds, peak Python MiB).

    Tracing is started fresh for each section; avoid nested calls to this helper.
    """
    tracemalloc.start()
    tracemalloc.reset_peak()
    start = time.perf_counter()
    try:
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, elapsed, peak / (1024 ** 2)


def model_size_kib(model):
    """Return estimated serialised model size in KiB (not its RAM usage)."""
    return len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)) / 1024


def save_profile(folder, name, *, training_s, prediction_s, train_peak_mib,
                 prediction_peak_mib, model_kib=None, n_train=None, n_test=None):
    """Write a one-row CSV documenting per-model profiling scope and units."""
    from pathlib import Path
    import pandas as pd
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    row = dict(Model=name, Training_Time_s=training_s,
               Test_Prediction_Time_s=prediction_s,
               Training_Peak_Python_MiB=train_peak_mib,
               Test_Prediction_Peak_Python_MiB=prediction_peak_mib,
               Model_Size_KiB=model_kib, Training_Rows=n_train, Test_Rows=n_test)
    pd.DataFrame([row]).to_csv(folder / f"{name}_performance.csv", index=False)
    performance_table(row)
    return row


def save_evaluations(name, model, splits, predict, folder, training_profile=None, training_rows=None):
    """Save train/validation/test scores, measured-only test scores, and test predictions."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    test_profile = None
    for label,part in zip(['Training','Validation','Testing'],splits):
        if label == 'Testing':
            prediction, predict_seconds, predict_peak = profile_call(predict, part)
            test_profile = (predict_seconds, predict_peak)
        else:
            prediction=predict(part)
        rows.append({'Dataset':label,**scores(part[TARGET],prediction)})
        if label=='Testing':
            output=pd.DataFrame({'target_time':part['target_time'].to_numpy(),
                   'actual_occupancy':part[TARGET].to_numpy(), 'predicted_occupancy':prediction})
            if 'target_is_measured' in part:
                output['target_is_measured']=part['target_is_measured'].to_numpy()
                mask=part['target_is_measured'].to_numpy(dtype=bool)
                if mask.sum()>=2:
                    rows.append({'Dataset':'Measured Targets Only',**scores(part[TARGET].to_numpy()[mask],prediction[mask])})
            output.to_csv(folder/f'{name}_predictions.csv',index=False)
    pd.DataFrame(rows).to_csv(folder/f'{name}_results.csv',index=False)
    scores_table(rows)

    if training_profile is not None and test_profile is not None:
        save_profile(folder, name, training_s=training_profile[0],
                     prediction_s=test_profile[0], train_peak_mib=training_profile[1],
                     prediction_peak_mib=test_profile[1],
                     model_kib=model_size_kib(model) if model is not None else None,
                     n_train=training_rows if training_rows is not None else len(splits[0]),
                     n_test=len(splits[2]))
