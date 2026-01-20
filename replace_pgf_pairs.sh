#!/bin/bash

# Usage: ./replace_from_pairs.sh input.pgf output.pgf replacements.txt

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 input.pgf output.pgf replacements.txt"
    exit 1
fi

input_file="$1"
output_file="$2"
replacements_file="$3"

# Copy the input file to the output file to start
cp "$input_file" "$output_file"

# Apply each replacement line from the replacements file
while IFS= read -r line; do
    # Skip empty lines or lines that don't contain a pair
    [[ "$line" =~ ^#.*$ || -z "$line" || ! "$line" =~ .+,[[:space:]]?.+ ]] && continue

    old=$(echo "$line" | cut -d',' -f1)
    new=$(echo "$line" | cut -d',' -f2-)

    # Escape slashes and ampersands in the new string
    escaped_new=$(printf '%s\n' "$new" | sed 's/[&/\]/\\&/g')

    # Replace in-place using sed
    sed -i "s/$old/$escaped_new/g" "$output_file"
done < "$replacements_file"

echo "Replacements applied to $output_file from $replacements_file"
