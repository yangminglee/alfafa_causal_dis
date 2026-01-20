#!/bin/bash

# Usage: ./replace_in_pgf.sh input.pgf output.pgf "old_string" "new_string"

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 input.pgf output.pgf 'old_string' 'new_string'"
    exit 1
fi

input_file="$1"
output_file="$2"
old_string="$3"
new_string="$4"

# Escape slashes and ampersands in new_string to avoid sed errors
escaped_new=$(printf '%s\n' "$new_string" | sed 's/[&/\]/\\&/g')

# Replace all occurrences
sed "s/$old_string/$escaped_new/g" "$input_file" > "$output_file"

echo "Replaced '$old_string' with '$new_string' in $output_file"
