# !pip install habanero
import pandas as pd
from habanero import Crossref
from tqdm import tqdm
from tqdm.contrib import tenumerate
import os
import json

# Initialize Crossref client
cr = Crossref()

def get_paper_data(doi_list, id_list, temp_dir="temp_data"):
  """
  Create a DataFrame containing paper titles and abstracts from a list of DOIs.

  Args:
    doi_list: List of DOIs
    id_list: List of COD IDs
    temp_dir: Directory for temporary file storage

  Returns:
    DataFrame containing titles and abstracts of papers
  """

  data = []
  os.makedirs(temp_dir, exist_ok=True)
  
  for i, (doi, cod_id) in tenumerate(zip(doi_list, id_list)):
    temp_filepath = os.path.join(temp_dir, f"{cod_id}.json")
    if os.path.exists(temp_filepath):
      with open(temp_filepath, 'r') as f:
        temp_data = json.load(f)
        data.append(temp_data)
      print(f"Skipping ID {cod_id} - response already cached.")
      continue

    try:
      # Retrieve metadata using the Crossref API
      res = cr.works(ids=doi)
      title = res['message']['title'][0] if 'title' in res['message'] else None
      abstract = res['message']['abstract'] if 'abstract' in res['message'] else None
      temp_data = {'DOI': doi, 'File': cod_id, 'Title': title, 'Abstract': abstract}

      # Convert int64 type data to int
      if 'File' in temp_data:
          temp_data['File'] = int(temp_data['File'])

      data.append(temp_data)
      with open(temp_filepath, 'w') as f:
        json.dump(temp_data, f)
    except Exception as e:
      print(f"Error fetching data for DOI: {doi}, Error: {e}")
      temp_data = {'DOI': doi, 'File': cod_id, 'Title': None, 'Abstract': None}

      # Convert int64 type data to int
      if 'File' in temp_data:
          temp_data['File'] = int(temp_data['File'])

      data.append(temp_data)
      with open(temp_filepath, 'w') as f:
        json.dump(temp_data, f)
  return pd.DataFrame(data)


def filter_dataframe(df):
  """
  Filter the DataFrame to keep only rows where both Title and Abstract are not None.

  Args:
    df: Input DataFrame

  Returns:
    Filtered DataFrame
  """

  return df.dropna(subset=['Title', 'Abstract'])


if __name__ == '__main__':
    cod_metadata = pd.read_csv("../data/cod_metadata_20250801.csv")
    save_root_path = "../data/cod_full_20250801"  

    doi_list = cod_metadata['doi'].values
    id_list = cod_metadata['file'].values
    
    df = get_paper_data(doi_list, id_list)
    filtered_df = filter_dataframe(df)
    filtered_df.to_csv(f"{save_root_path}/retrieved_abstract_20250801.csv")

    # output_lines = []

    # for i in range(len(filtered_df)):
    #     title_format = f"Title: {filtered_df.iloc[i]['Title']}"
    #     abst_format = f"Abstract: {filtered_df.iloc[i]['Abstract']}"
    #     cod_id = filtered_df.iloc[i]['File']
        
    #     output_lines.append(f"ID {cod_id}\n{title_format}\n{abst_format}\n")

    # # Write output to a text file
    # with open(f'{save_root_path}/title_abst_pair_all.txt', 'w') as file:
    #     file.writelines(output_lines)
