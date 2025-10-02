import json
import pandas as pd 

def create_dataframe_from_json_strings(json_strings):
  """
  Create a DataFrame from a list of JSON strings.

  Args:
    json_strings: A list of JSON strings.

  Returns:
    pandas.DataFrame: A DataFrame created from the JSON data.
                      Entries that do not contain both 'ID' and 'Keywords' keys will be skipped.
  """

  data = []
  for json_string in json_strings:
    try:
      json_data = json.loads(json_string.strip("`json\n"))
      for entry in json_data:
        if all(key in entry for key in ["ID", "Keywords"]):
          data.append([entry["ID"], entry["Keywords"]])
    except json.JSONDecodeError:
        # print("invalid skipped")
        pass
      # print(f"Invalid JSON string: {json_string}")
  return pd.DataFrame(data, columns=["ID", "Keywords"])

# Function to remove entries with empty keyword lists
def remove_empty_entries(df, column_name):
    return df[df[column_name].map(bool)]

def exclude_keywords(df, column_name, keywords_to_exclude):
    # Create a copy of the dataframe to avoid SettingWithCopyWarning
    df_copy = df.copy()

    # Transform to lowercase for case-insensitive comparison
    lowercase_excluded = {kw.lower() for kw in keywords_to_exclude}
    
    # Apply the exclusion process on each keyword list
    def clean_keywords(kw_list):
        return [kw for kw in kw_list if kw.lower() not in lowercase_excluded]

    df_copy[column_name] = df_copy[column_name].apply(clean_keywords)
    return df_copy