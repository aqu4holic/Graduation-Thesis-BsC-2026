# %pip install pytorch_lightning


# update the token via https://hub.crunchdao.com/competitions/causality-discovery/submit/via/notebook

#%pip install crunch-cli --upgrade --quiet --progress-bar off
#!crunch --env staging setup-notebook causality-discovery kpewOhlUGGNLMh0edTaDAs8p --no-data


#%env API_BASE_URL=https://api.hub.crunchdao.io/
#%env WEB_BASE_URL=https://hub.crunchdao.io/

import crunch.store
#crunch.store.load_from_env()


import typing
import os
from tqdm.auto import tqdm

# Common data science tools
import pandas as pd
import numpy as np

# PyTorch for building and training neural networks
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# We use PyTorch Lightning for training
import pytorch_lightning as pl

# NetworkX for working with graphs
import networkx as nx

# Scikit-learn for data splitting
from sklearn.model_selection import train_test_split


import crunch

#crunch = crunch.load_notebook()


class CausalDataset(Dataset):
    """
    A PyTorch Dataset class for handling causal discovery data.

    Attributes:
        X (np.ndarray): A 3D numpy array of shape (num_samples, 1000, 10) containing the input features.
        y (np.ndarray): A 3D numpy array of shape (num_samples, 10, 10) containing the target values.
        target_mask (np.ndarray): A 3D boolean numpy array of shape (num_samples, 10, 10) indicating the presence of target values.
    """

    def __init__(
        self,
        X: typing.List[pd.DataFrame],
        y: typing.List[pd.DataFrame]
    ) -> None:
        """
        Initializes the dataset with input features and target values.
        """

        # The shape of X is (num_samples, 1000, 10), where 1000 is number of rows and 10 is maximum number of variables
        self.X = np.zeros([len(X), 1000, 10], dtype=np.float32)

        # The shape of y is (num_samples, 10, 10), where 10 is the maximum number of variables
        self.y = np.zeros([len(X), 10, 10], dtype=np.float32)

        # The target mask is a boolean array indicating the presence of target values, it is need for model training because not all datasets have 10 variables
        self.target_mask = np.zeros([len(X), 10, 10], dtype=bool)

        for i in range(len(X)):
            self.X[i, :X[i].shape[0], :X[i].shape[1]] = X[i].values
            self.y[i, :y[i].shape[0], :y[i].shape[1]] = y[i].values
            self.target_mask[i, :y[i].shape[0], :y[i].shape[1]] = True

    def __len__(self) -> int:
        """
        Returns:
            The number of samples in the dataset.
        """

        return len(self.X)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves the sample at the specified index.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            A dictionary containing 'X', 'y', and 'target_mask' for the specified index.
        """

        X = self.X[idx]
        y = self.y[idx]
        target_mask = self.target_mask[idx]

        return {
            'X': X,
            'y': y,
            'target_mask': target_mask
        }


def preprocessing(X: pd.DataFrame):
    """
    Preprocesses the input data for neural network.

    Args:
        X: The input data as a pandas DataFrame.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: The input data converted to a PyTorch tensor and unsqueezed.
            - torch.Tensor: A mask tensor of ones with the same shape as the input tensor.
    """

    x = torch.Tensor(X.values).unsqueeze(0)
    return x


class CausalModel(nn.Module):
    """
    A neural network model for causal discovery.

    Attributes:
        input_layer (nn.Sequential): The input layer consisting of a linear layer, ReLU activation, and another linear layer.
        final (nn.Sequential): The final layer consisting of a linear layer, ReLU activation, and another linear layer.
    """

    def __init__(self, d_model=64):
        """
        Args:
            d_model: The dimension of the model. Default is 64.
        """

        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2 * d_model)
        )

        self.final = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        Args:
            x: The input tensor.

        Returns:
            The output tensor after applying the model.
        """

        # Compute the query and key tensors
        q, k = self.input_layer(x.unsqueeze(-1)).chunk(2, dim=-1)

        # Perform the scaled dot-product attention
        x = torch.einsum('b s i d, b s j d -> b i j d', q, k) * (x.shape[1] ** -0.5)

        y = self.final(x).squeeze(-1)
        return y


class ModelWrapper(pl.LightningModule):
    """
    A PyTorch Lightning Module wrapper for a causal model.

    Attributes:
        model (CausalModel): The causal model being wrapped.
        train_criterion (nn.BCEWithLogitsLoss): The loss function used for training, which is Binary Cross-Entropy with a class weight of 5.0 for the positive class.
    """

    def __init__(self, d_model=64):
        """
        Args:
            d_model: The dimension of the model. Default is 64.
        """

        super().__init__()

        self.model = CausalModel(d_model)

        # The loss function is Binary Cross-Entropy with a class weight of 5.0 for the positive class, to account for class imbalance.
        self.train_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the model.
        """

        return self.model(x)

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for training.
        """

        # We use the Adam optimizer with a learning rate of 1e-3.
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        # The learning rate is reduced by a factor of 0.1 after the 7th epoch.
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 7, gamma=0.1, last_epoch=-1)

        return [optimizer], [scheduler]

    def training_step(self, train_batch: dict, batch_idx: int):
        """
        Defines a single training step, including the computation of the loss and logging.
        """

        x = train_batch['X']
        y = train_batch['y']
        target_mask = train_batch['target_mask']

        preds = self(x)

        loss = self.train_criterion(preds[target_mask], y[target_mask])

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True
        )

        return loss


def transform_proba_to_DAG(
    nodes: typing.List[str],
    pred: np.ndarray
) -> np.ndarray:
    """
    Transforms a probability matrix into a Directed Acyclic Graph (DAG).

    Parameters:
        nodes: A list of node names.
        pred: A 2D numpy array representing the probability matrix.

    Returns:
        A 2D numpy array representing the adjacency matrix of the DAG.
    """

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edge('X', 'Y')

    x_index, y_index = np.unravel_index(np.argsort(pred.ravel())[::-1], pred.shape)
    for i, j in zip(x_index, y_index):
        n1 = nodes[i]
        n2 = nodes[j]

        if i == j:
            continue

        if ((n1 == 'X') and (n2 == 'Y')) or ((n1 == 'Y') and (n2 == 'X')):
            continue

        if pred[i, j] > 0.5:
            G.add_edge(n1, n2)

            if not nx.is_directed_acyclic_graph(G):
                G.remove_edge(n1, n2)

    G = nx.to_numpy_array(G)
    return G


def graph_nodes_representation(graph, nodelist):
    """
    Create an alternative representation of a graph which is hashable
    and equivalent graphs have the same hash.

    Python cannot PROPERLY use nx.Graph/DiGraph as key for
    dictionaries, because two equivalent graphs with just different
    order of the nodes would result in different keys. This is
    undesirable here.

    So here we transform the graph into an equivalent form that is
    based on a specific nodelist and that is hashable. In this way,
    two equivalent graphs, once transformed, will result in identical
    keys.

    So we use the following trick: extract the adjacency matrix
    (with nodes in a fixed order) and then make a hashable thing out
    of it, through tuple(array.flatten()):
    """

    # This get the adjacency matrix with nodes in a given order, as
    # numpy array (which is not hashable):
    adjacency_matrix = nx.adjacency_matrix(graph, nodelist=nodelist).todense()

    # This transforms the numpy array into a hashable object:
    hashable = tuple(adjacency_matrix.flatten())

    return hashable


def create_graph_label():
    """
    Create a dictionary from graphs to labels, in two formats.
    """

    graph_label = {
        nx.DiGraph([("X", "Y"), ("v", "X"), ("v", "Y")]): "Confounder",
        nx.DiGraph([("X", "Y"), ("X", "v"), ("Y", "v")]): "Collider",
        nx.DiGraph([("X", "Y"), ("X", "v"), ("v", "Y")]): "Mediator",
        nx.DiGraph([("X", "Y"), ("v", "X")]): "Cause of X",
        nx.DiGraph([("X", "Y"), ("v", "Y")]): "Cause of Y",
        nx.DiGraph([("X", "Y"), ("X", "v")]): "Consequence of X",
        nx.DiGraph([("X", "Y"), ("Y", "v")]): "Consequence of Y",
        nx.DiGraph({"X": ["Y"], "v": []}): "Independent",
    }

    nodelist = ["v", "X", "Y"]

    # This is an equivalent alternative to graph_label but in a form for which two equivalent graphs have the same key:
    adjacency_label = {
        graph_nodes_representation(graph, nodelist): label
        for graph, label in graph_label.items()
    }

    return graph_label, adjacency_label


def get_labels(adjacency_matrix, adjacency_label):
    """
    Transform an adjacency_matrix (as pd.DataFrame) into a dictionary of variable:label
    """

    result = {}
    for variable in adjacency_matrix.columns.drop(["X", "Y"]):
        submatrix = adjacency_matrix.loc[[variable, "X", "Y"], [variable, "X", "Y"]]  # this is not hashable
        key = tuple(submatrix.values.flatten())  # this is hashable and compatible with adjacency_label

        result[variable] = adjacency_label[key]

    return result


#X_train, y_train, X_test = crunch.load_data()


# Train test split
#train_keys, test_keys = train_test_split(list(X_train.keys()), test_size=0.2, random_state=42)

#print("Train datasets (top 5):", train_keys[:5])
#print("Test datasets (top 5):", test_keys[:5])

#X_train_split = [X_train[key] for key in train_keys]
#y_train_split = [y_train[key] for key in train_keys]
#X_test_split = [X_train[key] for key in test_keys]
#y_test_split = [y_train[key] for key in test_keys]

#train_dataset = CausalDataset(X_train_split, y_train_split)
#train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=4)

#test_dataset = CausalDataset(X_test_split, y_test_split)
#test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False, num_workers=4)

#print("Number of training samples:", len(train_dataset))
#print("Number of test samples:", len(test_dataset))


# Model Training
#model = ModelWrapper(d_model=64)
#trainer = pl.Trainer(accelerator="gpu", max_epochs=10, logger=True, enable_checkpointing=False, enable_progress_bar=True)
#trainer.fit(model, train_loader)


# Model Inference
#graph_label, adjacency_label = create_graph_label()

#model = model.eval()

#y_pred = []
#y_true = []
#for name in tqdm(test_keys):
#    X = X_train[name]
#    y = y_train[name]
#    x = preprocessing(X)
#
#    with torch.no_grad():
#        pred = model(x)[0]
#        pred = torch.sigmoid(pred)
#        pred = pred.cpu().numpy()
#
#    nodes = list(X.columns)
#    pred = transform_proba_to_DAG(nodes, pred).astype(int)
#    A = pd.DataFrame(pred, columns=nodes, index=nodes)
#
#    predicted_label = get_labels(A, adjacency_label)
#    ground_truth_label = get_labels(y, adjacency_label)
#
#    for key in predicted_label.keys():
#        y_pred.append(predicted_label[key])
#        y_true.append(ground_truth_label[key])

#y_pred = pd.Series(y_pred)
#y_true = pd.Series(y_true)


# Calculate Balanced Accuracy and Accuracy per class
#scores = {}

#for label in y_true.unique():
#    scores[label] = np.mean(y_pred[y_true == label] == label)

#scores = pd.Series(scores)
#scores['Balanced Accuracy'] = scores.mean()

#display(scores)


def train(
    X_train: typing.Dict[str, pd.DataFrame],
    y_train: typing.Dict[str, pd.DataFrame],
    # number_of_features: int,
    model_directory_path: str,
    # id_column_name: str,
    # prediction_column_name: str,
    # has_gpu: bool,
) -> None:
    X = []
    y = []
    for dataset_id in X_train:
        X.append(X_train[dataset_id])
        y.append(y_train[dataset_id])

    dataset = CausalDataset(X,y)
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=7)

    model = ModelWrapper(d_model=64)
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=10,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False
    )
    trainer.fit(model, train_dataloader)

    model_path_file = os.path.join(model_directory_path, "model.pt")
    torch.save(model.model.state_dict(), model_path_file)


# Uncomment what you need!
def infer(
    X_test: typing.Dict[str, pd.DataFrame],
    # number_of_features: int,
    model_directory_path: str,
    id_column_name: str,
    prediction_column_name: str,
    # has_gpu: bool,
    # has_trained: bool,
) -> pd.DataFrame:
    model_path_file = os.path.join(model_directory_path, "model.pt")

    model = CausalModel(d_model=64)
    model = model.eval()
    model.load_state_dict(torch.load(model_path_file, map_location='cpu'))

    submission_file = {}
    for name in X_test:
        X = X_test[name]
        x = preprocessing(X)

        with torch.no_grad():
            pred = model(x)[0]
            pred = torch.sigmoid(pred)
            pred = pred.cpu().numpy()

        nodes = list(X.columns)
        pred = transform_proba_to_DAG(nodes, pred).astype(int)
        G = pd.DataFrame(pred, columns=nodes, index=nodes)

        for i in nodes:
            for j in nodes:
                submission_file[f'{name}_{i}_{j}'] = int(G.loc[i,j])

    submission_file = pd.Series(submission_file)
    submission_file = submission_file.reset_index()
    submission_file.columns = [id_column_name, prediction_column_name]

    return submission_file


#crunch.test(
#    no_determinism_check=True
#)

#print("Download this notebook and submit it to the platform: https://hub.crunchdao.com/competitions/causality-discovery/submit/via/notebook")
