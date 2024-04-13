import csv
from config import POSES_OUTPUT_PATH

mp_path = POSES_OUTPUT_PATH / "ycbv" / "gt-refiner-final_ycbv-test.csv"
test_dir_path = POSES_OUTPUT_PATH / "ycbv" / "convex_hull_only" / "gravity-three-phase"

mp_lines = 0
deleted_files = 0
num_files = len(list(test_dir_path.iterdir()))

with open(mp_path, "r") as f:
    reader = csv.reader(f)
    mp_lines = len(list(reader))

for test_path in test_dir_path.iterdir():
    with open(test_path, "r") as f:
        reader = csv.reader(f)
        test_lines = len(list(reader))
    print(f"{test_path.name}: {test_lines}/{mp_lines}")
    if test_lines < mp_lines:
        deleted_files += 1
        print("^^^^^^ Deleted ^^^^^^")
        #test_path.rename(test_path.parent.parent / "corrupt" / test_path.name)
        test_path.unlink()
        
print(f"Deleted files: {deleted_files}/{num_files}")