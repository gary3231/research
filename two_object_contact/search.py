import numpy as np
import torch
from sklearn.neighbors import KDTree

def search(body_1: torch.Tensor, body_2: torch.Tensor) -> torch.Tensor:
    """
    Perform a search operation and return a 2D tensor.

    Args:
        possible contact points of body_1 (torch.Tensor): A 2D tensor input.
        all points on the contact surface of body_2 (torch.Tensor): A 2D tensor input.

    Returns:
         corresponding contact points of body_2 (torch.Tensor): A 2D tensor output.
    """
    
    kdtree = KDTree(body_2.cpu().detach().numpy())

    possible = torch.zeros(body_1.shape)

    i = 0
    for p in body_1.cpu().detach().numpy():
        p = np.reshape(p, (-1, 2))
        _, indices = kdtree.query(p, k=1)
        possible[i] = body_2[indices[0]]
        i+=1

    return possible
# Sample data as a PyTorch tensor
def test():
    data = torch.tensor([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]], dtype=torch.float32)

    # Convert PyTorch tensor to numpy array for KDTree
    # data_np = data.numpy()

    # Build KD tree
    kdtree = KDTree(data, leaf_size=30)

    # Query point as a PyTorch tensor
    query_point = torch.tensor([[7, 2]], dtype=torch.float32)

    # Convert query point to numpy for KDTree
    # query_point_np = query_point.numpy()

    # Query the KD tree for nearest neighbors
    distances, indices = kdtree.query(query_point, k=1)
    # a = torch.zeros(data.shape)
    print(indices)
    # Print results
    print("Query Point (Tensor):", query_point)
    print("Nearest Neighbors (from KDTree):")
    for i, idx in enumerate(indices[0]):
        neighbor = (data[idx])  # Convert back to tensor if needed
        print(f"Neighbor {i + 1}: {neighbor}, Distance: {distances[0][i]}")

if __name__ == "__main__":
    test()
