import csv
from config import POSES_OUTPUT_PATH

mp_path = POSES_OUTPUT_PATH / "tless" / "refiner-final-filtered_tless-test.csv"
test_dir_path = POSES_OUTPUT_PATH / "tless" / "final-fog"

mp_lines = 0
deleted_files = 0
duplicate_files = 0
num_files = len(list(test_dir_path.iterdir()))

def delete_dupe_lines(csv_in, csv_out):
    with open(csv_in, 'r') as in_file, open(csv_out, 'w') as out_file:
        seen = set() # set for fast O(1) amortized lookup
        for line in in_file:
            if ",".join(line.split(",")[:3]) in seen:
                continue # skip duplicate
            seen.add(",".join(line.split(",")[:3]))
            out_file.write(line)
    csv_out.rename(csv_in)

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
        #test_path.unlink()
    elif test_lines > mp_lines:
        duplicate_files += 1
        print("^^^^^^ Duplicate ^^^^^^")
        #delete_dupe_lines(test_path, test_path.parent / "temp.csv")
        
print(f"Deleted files: {deleted_files}/{num_files}")
print(f"Duplicate files: {duplicate_files}/{num_files}")