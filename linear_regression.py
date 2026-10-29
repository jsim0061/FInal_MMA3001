"""Fit an untuned linear regression comparator on the fixed room time split."""
import argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parent
from sklearn.linear_model import LinearRegression
from utils import FEATURES, TARGET, load_data, chronological_split, save_evaluations
from utils import profile_call


def main():
    """Train LinearRegression on past data and save all evaluation metrics."""
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=ROOT/'occupancy_15min_cleaned.csv')
    args=ap.parse_args()
    splits=chronological_split(load_data(args.data))
    model, fit_time, fit_peak = profile_call(
        lambda: LinearRegression().fit(splits[0][FEATURES], splits[0][TARGET]))
    save_evaluations('linear_regression',model,splits,lambda p:model.predict(p[FEATURES]),ROOT/'results', training_profile=(fit_time,fit_peak))
if __name__=='__main__':main()
