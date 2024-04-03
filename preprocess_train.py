# Read the contents of the original file
with open('set1.train.txt', 'r') as file:
    original_lines = file.readlines()

# Count the number of lines in the original file
original_length = len(original_lines)

# Filter out entries with relevance score 0
filtered_lines = [line.strip() for line in original_lines if float(line.split()[0]) != 0]

# Count the number of lines in the filtered file
filtered_length = len(filtered_lines)

# Write the filtered lines back to a new file
with open('train_filtered.txt', 'w') as file:
    file.write('\n'.join(filtered_lines))

print("Original length:", original_length)
print("Filtered length:", filtered_length)
print("Filtered file saved as 'train_filtered.txt'")
