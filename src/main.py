from generate_data import generate_dummy_data
from generate_data_noise import generate_noisy_dummy_data
from generate_data_year import generate_noisy_yearly_dummy_data
from cycle_detector import detect_cycles
from new_approach_derivative import detect_cycles_diff

if __name__ == "__main__":
    print("Generating dummy data...")
    generate_noisy_yearly_dummy_data("data/dummy_mwsel_year.csv")

    print("Detecting cycles...")
    detect_cycles_diff(
        csv_path="data/dummy_mwsel_year.csv",
        db_path="cycle_results_year.db"
    )

    print("Pipeline finished successfully")
