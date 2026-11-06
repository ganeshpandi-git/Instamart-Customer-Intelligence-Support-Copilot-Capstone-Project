import os

def validate_artifacts():
    expected_files = [
        "data/instamart.db",
        "outputs/customer_segments.csv",
        "outputs/lapse_model.joblib",
        "outputs/retention_list.csv",
        "outputs/triage_results.csv",
        "outputs/part_d/rag_eval_results.csv"
    ]
    
    print("--- Instamart Capstone Project Artifact Validation ---")
    all_present = True
    for f in expected_files:
        exists = os.path.exists(f)
        status = "FOUND" if exists else "MISSING"
        print(f"[{status}] {f}")
        if not exists:
            all_present = False
            
    if all_present:
        print("\n🎉 All core project artifacts are successfully generated and verified!")
    else:
        print("\n⚠️ Some artifacts are missing. Please check previous steps.")

if __name__ == "__main__":
    validate_artifacts()