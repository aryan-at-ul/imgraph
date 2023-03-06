from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data


def get_train_test_loader(train_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
    return train_loader, test_loader