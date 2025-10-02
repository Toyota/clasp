# !pip install -U vllm
import os

# Set HuggingFace token (optional)
# with open('../api_keys/hf_token.txt', mode="r") as f:
#     os.environ["HF_TOKEN"] = f.read()
#     os.environ["HUGGING_FACE_HUB_TOKEN"] = f.read()

import os
import json
import pandas as pd
from time import sleep
from tqdm import tqdm
from vllm import LLM, SamplingParams
import shutil


def prompt_format_func(material_id, title, abstract):
    """Formats the prompt for the Gemini model."""
    prompt_template = """Below are title-abstract pairs for materials science papers dealing with crystal structures. For each paper, list up to 10 keywords in English that describe the features, functions, or applications of the material discussed. Focus on the material itself, and do not include general terms or measurement techniques (e.g., Crystal Structure, Crystal Lattice, X-ray diffraction, Neutron Diffraction, Powder Diffraction). Return the results in json format with the following schema.

    **Example input 1:**

    ```
    ID: 0001
    Title: Enhancement of Critical Temperature in Layered Copper Oxide Superconductors via Lattice Compression Techniques
    Abstract: Superconductivity in copper oxides (cuprates) offers vast potential for technological applications due to their high critical temperatures (Tc). Our research presents a novel approach to enhance Tc in layered cuprate materials through the controlled application of lattice compression. Using advanced crystallographic methods, we systematically altered the interlayer spacing and analyzed the resultant changes in electronic properties. Our findings demonstrate a significant improvement in superconducting behavior at elevated temperatures, further supporting the unconventional mechanisms underpinning superconductivity in these materials. 
    ```

    **Example output 1:**

    ```json
    [  {
        "ID": "0001",
        "Keywords": [
          "High-Tc",
          "Cuprate Superconductors",
          "Lattice Compression",
          "Electronic Properties",
          "Layered Structures",
          "Superconducting Phase",
          "Temperature Enhancement",
          "Unconventional Superconductivity"
        ]
      }]
    ```

    **Example input 2:**

    ```
    ID: 0002
    Title: Advancements in Biodegradable Polymers for Sustained Drug Delivery Systems
    Abstract: The development of biocompatible and biodegradable materials is critical in the field of medical implants and drug delivery systems. This paper examines the latest advancements in biodegradable polymers tailored for sustained release of therapeutic agents. We analyze various polymer compositions that provide controlled degradation rates and compatibility with a range of drugs. Our results show promising applications in long-term treatments, reducing the need for repeated administration and improving patient compliance.
    ```

    **Example output 2:**

    ```json
    [  {
        "ID": "0002",
        "Keywords": [
          "Biomaterials",
          "Biodegradable Polymers",
          "Sustained Release",
          "Drug Delivery Systems",
          "Biocompatibility",
          "Controlled Degradation",
          "Therapeutic Agents",
          "Medical Implants",
          "Long-Term Treatment"
        ]
      }]
    ```
    """
    prompt = prompt_template + f"""
    **Input :**

    ```
    ID: {material_id}
    Title: {title}
    Abstract: {abstract} 
    ```

    **Output :**

    ```json
    """
    return prompt
# Define chunk size for processing
CHUNK_SIZE = 500

def process_chunk(prompts, mat_ids, llm, sampling_params, chunk_index, output_dir, model_id):
    """Processes a chunk of prompts and saves the results to a JSON file.

    Args:
        prompts (list): A list of prompts.
        mat_ids (list): A list of material IDs corresponding to the prompts.
        llm (vllm.LLM): The LLM object.
        sampling_params (vllm.SamplingParams): Sampling parameters for the LLM.
        chunk_index (int): The index of the current chunk.
        output_dir (str): The directory to save the chunk file.
        model_id (str): The ID of the language model used.
    """
    print(f"Processing chunk {chunk_index} with model {model_id}...")
    llm_outputs = llm.generate(prompts, sampling_params)
    generated_data = {"prompt": [out.prompt for out in llm_outputs],
                      "ID": mat_ids,
                      }
    for i in range(0, len(llm_outputs[0].outputs)):
        generated_data[f"output_{i}"] = [out.outputs[i].text for out in llm_outputs]

    # Create output directory if it doesn't exist
    model_output_dir = os.path.join(output_dir, model_id.replace('/', '_'))
    os.makedirs(model_output_dir, exist_ok=True)

    save_filename = f"chunk_{chunk_index}.json"
    save_path = os.path.join(model_output_dir, save_filename)
    with open(save_path, "w") as f:
        json.dump(generated_data, f)

def combine_results(num_chunks, model_id, output_dir):
    """Combines the results from multiple chunk files into a single JSON file.

    Args:
        num_chunks (int): The number of chunks that were processed.
        model_id (str): The ID of the language model used.
        output_dir (str): The directory containing the chunk files.
    """
    all_data = []
    model_output_dir = os.path.join(output_dir, model_id.replace('/', '_'))
    for i in range(num_chunks):
        chunk_filename = f"chunk_{i}.json"
        chunk_path = os.path.join(model_output_dir, chunk_filename)
        with open(chunk_path, "r") as f:
            chunk_data = json.load(f)
        all_data.append(chunk_data)
    
    # Combine the data from all chunks
    combined_data = {}
    for key in all_data[0].keys():
        combined_data[key] = [item for sublist in [d[key] for d in all_data] for item in sublist]
    
    save_filename = f"{model_id.replace('outputs/', '').replace('/', '').replace('.', '_')}.json"
    save_path = os.path.join(output_dir, f"cod_20250801_{save_filename}.json")
    with open(save_path, "w") as f:
        json.dump(combined_data, f)

    # Clean up chunk files and model directory
    for i in range(num_chunks):
        chunk_filename = f"chunk_{i}.json"
        chunk_path = os.path.join(model_output_dir, chunk_filename)
        os.remove(chunk_path)
    
    shutil.rmtree(model_output_dir)

if __name__ == "__main__":
    # Load retrieved abstruct data
    abst_df = pd.read_csv("../data/retrieved_abstract_20250801.csv")
    abst_df.rename(columns={"File": "material_id"}, inplace=True)
    abst_df.drop(columns=["Unnamed: 0"], inplace=True)

    model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    
    # 4gpus (tensor_parallel_size=4)
    llm = LLM(model=model_id, 
              tensor_parallel_size=4,
              enable_prefix_caching=False, 
              gpu_memory_utilization=0.90, 
              max_model_len=8192, 
              trust_remote_code=True,
              max_num_seqs=128)

    
    prompts = []
    mat_ids = []
    for i in abst_df.index:
        material_id = abst_df.loc[i]['material_id']
        material_id = str(material_id) 
        title = abst_df.loc[i]['Title']
        abstract = abst_df.loc[i]['Abstract']
        prompts.append(prompt_format_func(material_id, title, abstract))
        mat_ids.append(material_id)

    sampling_params = SamplingParams(n=5, temperature=0.5, top_p=0.95, max_tokens=1000)

    # Specify output directory (create if it doesn't exist)
    output_dir = "generated_data" 
    os.makedirs(output_dir, exist_ok=True)

    # Process data in chunks
    num_chunks = (len(prompts) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for i in tqdm(range(num_chunks)):
        start_idx = i * CHUNK_SIZE
        end_idx = min(start_idx + CHUNK_SIZE, len(prompts))
        process_chunk(prompts[start_idx:end_idx], 
                       mat_ids[start_idx:end_idx], 
                       llm, 
                       sampling_params, 
                       i,
                       output_dir,
                       model_id)

    # Combine results from all chunks
    combine_results(num_chunks, model_id, output_dir)