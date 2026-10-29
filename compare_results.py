"""Combine test-set metric CSVs produced by the four scripts."""
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent
from utils import section, table
root=Path(__file__).resolve().parent
items=[('Decision Tree',root/'decision_tree_results.csv'),('Historical Average',root/'results/historical_average_results.csv'),('Linear Regression',root/'results/linear_regression_results.csv'),('SVR',root/'results/svr_results.csv')]
rows=[]
for label,file in items:
    if not file.exists():
        print(f'Skipping {label}: missing {file}')
        continue
    frame=pd.read_csv(file);col='Dataset' if 'Dataset' in frame.columns else 'split'
    subset=frame[frame[col].astype(str).str.lower().eq('testing')]
    if not subset.empty:
        score=subset.iloc[0]
        row={'Model':label,'MAE':float(score['MAE']),'RMSE':float(score['RMSE']),'R2':float(score['R2'])}
        profile_name={'Decision Tree':'decision_tree', 'Historical Average':'historical_average',
                      'Linear Regression':'linear_regression', 'SVR':'svr'}[label]
        performance_file=root/'results'/f'{profile_name}_performance.csv'
        if performance_file.exists():
            performance=pd.read_csv(performance_file).iloc[0].to_dict()
            row.update({k:v for k,v in performance.items() if k != 'Model'})
        rows.append(row)
output=pd.DataFrame(rows)
(root/'results').mkdir(exist_ok=True)
output.to_csv(root/'results/model_comparison.csv',index=False)
if output.empty:
    print('No model results found. Run the model scripts first.')
else:
    section('MODEL COMPARISON  |  TEST DATA')
    table(['Model', 'MAE (people)', 'RMSE (people)', 'R²'], [
        [r['Model'], f"{r['MAE']:.3f}", f"{r['RMSE']:.3f}", f"{r['R2']:.3f}"]
        for _, r in output.iterrows()], numeric_columns=(1, 2, 3))
    profile_cols = ['Training_Time_s', 'Test_Prediction_Time_s',
                    'Training_Peak_Python_MiB', 'Test_Prediction_Peak_Python_MiB']
    if all(col in output.columns for col in profile_cols):
        section('COMPUTATIONAL PERFORMANCE  |  ALL MODELS')
        table(['Model', 'Train (s)', 'Predict (s)', 'Train peak (MiB)', 'Predict peak (MiB)'], [
            [r['Model']] + [f"{r[col]:.3f}" if pd.notna(r[col]) else 'N/A'
                              for col in profile_cols]
            for _, r in output.iterrows()], numeric_columns=(1, 2, 3, 4))
        print('  Peak memory is Python-tracked allocations, not total RAM.')
    print('\n  Full-precision results saved to results/model_comparison.csv')
