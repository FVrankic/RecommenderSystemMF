# Path to the original MovieLens 1M dataset
original_file = 'datasets/ml-1m/users.dat'
# Path to save the modified dataset
modified_file = 'datasets/ml-1m/ml1m_users.data'

# Open the original file, read content, and replace "::" with "\t"
with open(original_file, 'r') as file:
    content = file.read().replace('::', '\t')

# Write the modified content to a new file
with open(modified_file, 'w') as new_file:
    new_file.write(content)

print(f"Dataset has been saved with tab delimiters to {modified_file}")
