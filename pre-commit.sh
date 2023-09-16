#!/bin/sh

MAX_SIZE=5000000 # 5 MBs 
DIRECTORIES=("submission/" "notebooks/")  # Add more directories here

# Initialize an associative array to hold the directory sizes
declare -A dir_sizes
for dir in "${DIRECTORIES[@]}"; do
    dir_sizes["$dir"]=0
done

# Iterate over each changed file.
git diff --cached --name-only | while read -r file; do
    for dir in "${DIRECTORIES[@]}"; do
        if [[ "$file" == $dir* ]]; then
            # Accumulate the size of each file in this directory.
            size=$(git ls-files -s "$file" | awk '{print $4}')
            dir_sizes["$dir"]=$((${dir_sizes["$dir"]} + $size))
        fi
    done
done

# Check if the total size for any directory exceeds the maximum allowed size
for dir in "${!dir_sizes[@]}"; do
    if [ "${dir_sizes["$dir"]}" -gt "$MAX_SIZE" ]; then
        echo "Error: Total size of files in directory $dir is larger than allowed $MAX_SIZE bytes."
        exit 1
    fi
done
