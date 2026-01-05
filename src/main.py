from generate_data import generate_dummy_data
from cycle_detector import detect_cycles

if __name__ == "__main__":
    print("Generating dummy data...")
    generate_dummy_data("data/dummy_mwsel.csv")

    print("Detecting cycles...")
    detect_cycles(
        csv_path="data/dummy_mwsel.csv",
        db_path="cycle_results.db"
    )

    print("Pipeline finished successfully")
