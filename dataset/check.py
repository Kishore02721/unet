import os

# Define your three folder paths
folder1 = "original"
folder2 = "mask"
folder3 = "inspect"

# Get sets of filenames in each folder
files1 = set(os.listdir(folder1))
files2 = set(os.listdir(folder2))
files3 = set(os.listdir(folder3))

# Find the union of all filenames across folders
all_files = files1 | files2 | files3

# Find missing files in each folder
missing_in_f1 = all_files - files1
missing_in_f2 = all_files - files2
missing_in_f3 = all_files - files3

# Print missing files
if missing_in_f1 or missing_in_f2 or missing_in_f3:
    print("Missing files detected:")
    if missing_in_f1:
        print(f"Missing in {folder1}: {missing_in_f1}")
    if missing_in_f2:
        print(f"Missing in {folder2}: {missing_in_f2}")
    if missing_in_f3:
        print(f"Missing in {folder3}: {missing_in_f3}")
else:
    print("All folders have the same image names.")

