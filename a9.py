# python code to read data from two sources and check for exact matches and similarities
import warnings
import pandas as pd
from fuzzywuzzy import fuzz
warnings.filterwarnings('ignore')

source1 = {'id': [1, 2, 3, 4, 5],
          'name': ['John Doe', 'Jane Smith', 'Alice Johnson', 'Bob Brown', 'Charlie Black'],
          'address': ['123 Elm St', '456 Oak St', '789 Pine St', '101 Maple St', '202 Birch St']}

source2 = {'id': [1, 2, 3, 4, 5],
           'name': ['John Doe', 'Jane Smith', 'Alice Johnson', 'Bob Brown', 'Charlie Black'],
           'address': ['123 Elm St', '456 Oak St', '789 Pine St', '101 Maple St', '202 Birch Street']}

source1 = pd.DataFrame(source1)
source2 = pd.DataFrame(source2)

print(source1.head())
print(source2.head())


# Function to check for exact matches and similarities
def check_exact_matches(source1, source2):
    # Check for exact matches
    exact_matches = pd.merge(source1, source2, on=['id', 'name', 'address'], how='inner')
    return exact_matches


def check_similarities(source1, source2, threshold=80):
    # Check for similarities using fuzzy matching
    similarities = []
    for index1, row1 in source1.iterrows():
        for index2, row2 in source2.iterrows():
            name_similarity = fuzz.ratio(row1['name'], row2['name'])
            address_similarity = fuzz.ratio(row1['address'], row2['address'])

            if name_similarity > threshold and address_similarity > threshold:
                similarities.append({'id_source1': row1['id'], 'id_source2': row2['id'],
                                     'name_similarity': name_similarity,
                                     'address_similarity': address_similarity})
    return pd.DataFrame(similarities)


# Check for exact matches and similarities
exact_matches = check_exact_matches(source1, source2)
similarities = check_similarities(source1, source2)
print("Exact Matches:")
print(exact_matches)
print("Similarities:")
print(similarities)

# Save results to Excel
filename = 'output.xlsx'

with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
    exact_matches.to_excel(writer, sheet_name='Exact Matches', index=False)
    similarities.to_excel(writer, sheet_name='Similarities', index=False)
