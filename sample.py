import random

# Specify the input and output file paths
input_file = 'train_filtered.txt'
output_file = 'train_sampled.txt'

# Read all lines from the input file
with open(input_file, 'r') as f:
    lines = f.readlines()

# Calculate the number of lines to sample (1% of total lines)
num_lines = len(lines)
num_sample = int(num_lines * 0.01)

# Randomly sample the line indices
sample_indices = random.sample(range(num_lines), num_sample)

# Sort the sampled indices to maintain original order
sample_indices.sort()

# Write the sampled lines to the output file in the original order
with open(output_file, 'w') as f:
    for index in sample_indices:
        f.write(lines[index])

print(f"Sampled {num_sample} lines from {input_file} to {output_file} in original order.")
