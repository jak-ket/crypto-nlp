import pandas as pd

file_name = input("Please enter human eval file path: ")

df = pd.read_csv(file_name)

def iterate_dataframe(df):
    for index, row in df.iterrows():
        print(f">>> Text #{index + 1}: ", row['selftext'])
        user_input = input("\n>>> Enter label +1 (pos), 0 (neu), -1 (neg): ")
        df.at[index, 'human_label'] = int(user_input)
        print()
        df.to_csv(f"{file_name.split('.')[0]}_labeled.csv", index=False)

# Iterate over the DataFrame
iterate_dataframe(df)

# Display the updated DataFrame
print("Updated DataFrame:")
print(df)    