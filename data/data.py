from torch_geometric.datasets import Planetoid

# Download and load the Cora dataset
dataset = Planetoid(root='data', name='Cora')

# Access the graph data
data = dataset[0]

print(data)

