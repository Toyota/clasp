FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel
RUN pip install --upgrade pip \
&& pip install pandas tqdm jupyter jupyterlab ipython lightning pymatgen scipy scikit-learn transformers accelerate ConfigArgParse datasets tokenizers torchtext mlflow seqeval tensorboard mysql-connector-python hydra-core mp-api opentsne vllm
RUN conda install pyg -c pyg -y
RUN conda install -c pytorch -c nvidia faiss-gpu=1.8.0