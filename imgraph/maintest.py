from imgraph.pipeline import create_graph_pipleline
from imgraph.pipeline import create_graph_pipleline,load_saved_datasets
from imgraph.dataloader import get_train_test_loader
from imgraph.models import gat_model
from imgraph.utils import feature_size
import torch

# path = "/Users/aryansingh/projects/image_segmentation/chest_xray"

# create_graph_pipleline(path, 'classification', 'rag', 'resnet18', 10)


train_dataset, test_dataset = load_saved_datasets('pneumonia')

train_loader, test_loader = get_train_test_loader(train_dataset, test_dataset, 64)

feature_extractor = 'resnet18'

model = gat_model(feature_size[feature_extractor], 2)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        data.y = torch.Tensor(data.y)
        data.y = torch.Tensor(torch.flatten(data.y))
        data.y = data.y.type(torch.LongTensor)
        loss = criterion(out, data.y)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)  
         data.y = torch.Tensor(data.y)
         pred = out.argmax(dim=1).view(-1,1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.

for epoch in range(1, 11):
    train()
    try:
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    except Exception as e:
        print("error",e)
        pass





