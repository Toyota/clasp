import torch
import numpy as np

# for analysis embeddings
def search_kNN(embedding_query, embedding_target, use_gpu=False, k=10):
    import faiss
    """
    embeddingはtensorなどarray-like、kは探す近傍点の個数（1以上のint）
    return:
        D: クエリベクトルからk近傍までの距離
        I: クエリベクトルに対するk近傍のインデックス
    """
    vec_dim = embedding_target.shape[1]                   # ベクトルの次元(dimension)
    n_data = embedding_target.shape[0]                    # データベースのサイズ(database size)
    # x_target_vec = embedding_target.numpy().astype('float32')
    # x_query_vec = embedding_query.numpy().astype('float32')
    x_target_vec = embedding_target.to(dtype=torch.float32).numpy()
    x_query_vec = embedding_query.to(dtype=torch.float32).numpy()

    faiss_index_cpu = faiss.IndexFlatL2(vec_dim)
    if use_gpu:
        faiss_index_gpu = faiss.index_cpu_to_all_gpus(faiss_index_cpu)
        faiss_index_gpu.add(x_target_vec)
        D, I = faiss_index_gpu.search(x_query_vec, k)
    else:
        faiss_index_cpu.add(x_target_vec)
        D, I = faiss_index_cpu.search(x_query_vec, k)
    return D, I


def get_bool_of_corrected_predictions(embedding_query, embedding_target, k=10):
    D, I = search_kNN(embedding_query=embedding_query, embedding_target=embedding_target, k=k)
    series_idx = np.arange(embedding_target.shape[0]) # クエリベクトル自身のインデックス
    is_predict_correctly = np.isin(series_idx,  I[:, :k])
    return is_predict_correctly


def calc_grobal_top_k_acc(embedding_query, embedding_target, k=10):
    # top-k accを計算
    top_k_correct_samplenums = []
    top_k_acc = []
    n_data = embedding_target.shape[0]                    # データベースのサイズ(database size)
    for i in range(k):
        # k近傍のインデックスのarray内にクエリベクトル自身が入っていればok
        correct_samplenum = get_bool_of_corrected_predictions(embedding_query=embedding_query, 
                                                              embedding_target=embedding_target, 
                                                              k=i+1).sum() 
        correct_samplenum = correct_samplenum.astype(float)
        top_k_correct_samplenums.append(correct_samplenum)
        top_k_acc.append(correct_samplenum/n_data)
    return top_k_acc


def large_cdist(x, y, p=2, k=8, compute_mode='donot_use_mm_for_euclid_dist'):
    """
    Computes pairwise distance between every column between x and y,
    with an option of computing it on different devices.

    Args:
    - x: tensor of shape (n_batch_x, n_dim)
    - y: tensor of shape (n_batch_y, n_dim)
    - p: int, the desired degree of norm for the Minkowski metric used to compute distances.
    - k: int, Number of chunks to split `x` prior to sending them to cdist().
    - compute_mode: str, specifies how cross products are computed. 

    Returns:
    - dist: tensor of pairwise distances of shape (n_batch_x, n_batch_y)

    """

    # determine the dimensions of X and Y
    n_batch_x = x.shape[0]
    n_batch_y = n_batch_x
    n_dim = x.shape[1]
    
    # ensure the number of rows is divisible by k
    f = -n_batch_x % k
    if f > 0:
        z = torch.zeros((f, n_dim), dtype=x.dtype, device=x.device)
        x = torch.cat((x, z), dim=0)
        n_batch_x += f

    # restructure X and Y into k pieces
    x_ = x.reshape(k, n_batch_x//k, n_dim)
    y_ = y.reshape(1, n_batch_y, n_dim)

    # compute pairwise distance between X and Y
    dist = [torch.cdist(x_[i:i+1], y_, p=p, compute_mode=compute_mode) for i in range(k)]
    dist = torch.cat(dist,dim=0)
    dist = dist.reshape(n_batch_x, n_batch_y)

    # remove added last rows if it applies.
    if f > 0:
        dist = dist[:-f, :]

    return dist


def batch_wise_accuracy(x, y):
    """
    Within each mini-batch, this function performs cross-modal nearest neighbor search for each XRD embedding and crystal embedding, and evaluates the retrieval accuracy.
    If the embedding of the crystal corresponding to the queried XRD embedding is the nearest neighbor, it is considered as correct. 
    To do this, this function counts the rows in which the diagonal component of the distance matrix is the minimum in row-wise.

    Example:
    pairwise_dist_mat =
    tensor([[10.2470, 12.2066,  6.3246,  7.6811,  7.2801],
            [ 9.0000,  8.0623,  6.6332, 11.0905,  6.0828],
            [ 2.4495,  3.7417,  7.6811,  4.8990, 10.2956],
            [ 7.0711,  7.2111,  2.2361,  5.0990,  9.4868],
            [ 5.3852,  7.2801,  8.8318,  8.7750,  4.1231]])
    In case of this, the result is:
    batch_wise_pair_correct = tensor([0, 0, 0, 0, 1], dtype=torch.uint8)

    Parameters
    ----------
    x : torch.tensor (xrd embedding)
    y : torch.tensor (crystal embedding)

    Returns
    -------
    batch_wise_pair_correct: int
    batch_wise_pair_accuracy : float
    """

    pairwise_dist_mat = torch.cdist(x, y, p=2)
    positive_dist = pairwise_dist_mat.diag()
    n_sample = pairwise_dist_mat.shape[0]
    inf_arr = torch.ones(n_sample).type_as(x) * float("Inf")
    # Replace the distance of the diagonal components (=pair XRD and crystal) in the distance matrix with inf
    negative_pairwise_dist_mat = pairwise_dist_mat + torch.diag(inf_arr)
    # Calculate the percentage of rows for which the diagonal components of the distance matrix are row-wise minimums.
    batch_wise_pair_correct = positive_dist < negative_pairwise_dist_mat.min(1)[0]
    batch_wise_pair_accuracy = torch.sum(batch_wise_pair_correct).float()/n_sample
    return batch_wise_pair_correct, batch_wise_pair_accuracy
