import torch
import torch.nn.functional as F


def nt_xent_loss(embeddings, temperature=0.07):
    """
    Calculates the Normalized Temperature-scaled Cross Entropy Loss (NT-Xent Loss) for contrastive learning.

    Args:
        embeddings (torch.Tensor): Embeddings of shape (batch_size * 2, embedding_dim) for positive pairs within each row.
                                  The first half of the batch and the second half are positive pairs.
        temperature (float): Temperature scaling parameter.

    Returns:
        torch.Tensor: NT-Xent loss value (scalar).
    """
    batch_size = embeddings.shape[0] // 2
    embeddings_normalized = F.normalize(embeddings, dim=1)  # L2 normalize embeddings

    # Similarity matrix: pairwise cosine similarities
    similarity_matrix = torch.matmul(embeddings_normalized, embeddings_normalized.T)

    # Mask for positive pairs (within each batch row, i-th and (i+batch_size)-th are positives)
    positive_mask = torch.zeros((batch_size * 2, batch_size * 2), dtype=bool)
    for i in range(batch_size):
        positive_mask[i, i + batch_size] = positive_mask[i + batch_size, i] = True

    # Negative mask (all except positive pairs and self-similarity)
    negative_mask = ~positive_mask & ~torch.eye(batch_size * 2, dtype=bool)

    # Numerator of NT-Xent loss: similarity of positive pairs
    positive_similarities = similarity_matrix[positive_mask].view(batch_size, -1)

    # Denominator of NT-Xent loss: sum of similarities of negative pairs
    negative_similarities = similarity_matrix[negative_mask].view(batch_size, -1)
    denominator = torch.exp(negative_similarities / temperature).sum(dim=1, keepdim=True) + torch.exp(
        positive_similarities / temperature)

    # NT-Xent loss calculation
    nll = - (positive_similarities / temperature - torch.log(denominator))
    loss = nll.mean()

    return loss
