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


# Function to check for not matching
def check_no_matches(source1, source2):
    # Find rows in source1 that are not in source2
    no_matches_source1 = source1.merge(source2, on=['id', 'name', 'address'], how='left', indicator=True)
    no_matches_source1 = no_matches_source1[no_matches_source1['_merge'] == 'left_only'].drop(columns=['_merge'])

    # Find rows in source2 that are not in source1
    no_matches_source2 = source2.merge(source1, on=['id', 'name', 'address'], how='left', indicator=True)
    no_matches_source2 = no_matches_source2[no_matches_source2['_merge'] == 'left_only'].drop(columns=['_merge'])

    return pd.DataFrame(pd.concat([no_matches_source1, no_matches_source2]))


def check_similarities(source1, source2, threshold=80):
    # Check for similarities using fuzzy matching
    similarities = []
    for index1, row1 in source1.iterrows():
        for index2, row2 in source2.iterrows():
            name_similarity = fuzz.ratio(row1['name'], row2['name'])
            address_similarity = fuzz.ratio(row1['address'], row2['address'])

            if name_similarity > threshold and address_similarity > threshold and (name_similarity != 100 or address_similarity != 100):
                similarities.append({'id_source1': row1['id'],
                                     'id_source2': row2['id'],
                                     'name_similarity': name_similarity,
                                     'address_similarity': address_similarity})
    return pd.DataFrame(similarities)


# Check for exact matches and similarities
exact_matches = check_exact_matches(source1, source2)
similarities = check_similarities(source1, source2)
no_match = check_no_matches(source1, source2)
print("Exact Matches:")
print(exact_matches)
print("Similarities:")
print(similarities)
print('No Matches:')
print(no_match)

