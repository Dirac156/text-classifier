"""_summary_"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt


# Move model to device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIR = os.path.dirname(os.path.realpath(__file__))

SEQ_LENGTH = 250
TRAIN_DIR = os.path.join(DIR, "../data/train")
TEST_DIR = os.path.join(DIR, "../data/test")
EMBEDDING_PATH = os.path.join(DIR, "../data/all.review.vec.txt")
VOCAB = None


# Step 1: Load the reviews and labels
def load_reviews(data_dir: str) -> tuple[np.ndarray[str], np.ndarray[np.int32]]:
    """
    Load reviews and labels from the given data directory.

    Args:
        data_dir (str): The path to the data directory.

    Returns:
        reviews (np.ndarray[str]): The reviews.
        labels (np.ndarray[np.int32]): The labels.
    """
    reviews_list = []
    labels_list = []
    for label_type in ["positive", "negative"]:
        label_dir = os.path.join(data_dir, label_type)
        label = 1 if label_type == "positive" else 0
        for review_file in os.listdir(label_dir):
            file_path = os.path.join(label_dir, review_file)
            with open(file_path, "r", encoding="utf-8") as f:
                review = f.read().strip()
                reviews_list.append(review)
                labels_list.append(label)
    return (
        np.array(reviews_list, dtype=np.str_),
        np.array(labels_list, dtype=np.int32),
    )


# Step 2: Tokenize the reviews and create a vocabulary
def tokenize_reviews(reviews: np.ndarray[str]) -> np.ndarray[np.ndarray[str]]:
    """
    Split the input reviews into individual words.

    Args:
        reviews (np.ndarray[str]): The input reviews.

    Returns:
        np.ndarray[np.ndarray[str]]: The tokenized reviews.
    """
    # Split each review into individual words
    tokenized = [review.split() for review in reviews]

    # Convert the list of lists to an array of objects
    return np.array(tokenized, dtype=object)


def build_vocabulary(
    tokenized_reviews: np.ndarray[np.ndarray[str]], min_frequency: int = 1
) -> dict[str, int]:
    """
    Builds a vocabulary from the given tokenized reviews.

    Args:
        tokenized_reviews (np.ndarray[np.ndarray[str]]): The tokenized reviews.
        min_frequency (int, optional): The minimum frequency of a word to be included in the vocabulary. Defaults to 1.

    Returns:
        dict[str, int]: The vocabulary dictionary, where the keys are the words and the values are their corresponding indices.
    """
    word_frequencies: dict[str, int] = {}
    vocabulary: dict[str, int] = {}

    # Count word frequencies
    for review in tokenized_reviews:
        for word in review:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1

    # Create vocab with words above min frequency
    for word, frequency in word_frequencies.items():
        if frequency >= min_frequency:
            vocabulary[word] = (
                len(vocabulary) + 1
            )  # Start index from 1 (reserve 0 for padding)

    return vocabulary


# Step 3: Convert tokenized reviews to sequences of indices using NumPy
def reviews_to_indices(
    tokenized_reviews: np.ndarray[np.ndarray[str]],
    vocab: dict[str, int],
    max_len: int = 500,
) -> np.ndarray[np.ndarray[np.int32]]:
    """
    Convert tokenized reviews to sequences of indices using NumPy.

    Args:
        tokenized_reviews (np.ndarray[np.ndarray[str]]): The tokenized reviews.
        vocab (dict[str, int]): The vocabulary dictionary, where the keys are the words and the values are their corresponding indices.
        max_len (int, optional): The maximum length of the sequence. Defaults to 500.

    Returns:
        np.ndarray[np.ndarray[np.int32]]: The sequences of indices.
    """
    review_indices = np.zeros((len(tokenized_reviews), max_len), dtype=np.int32)

    for i, review in enumerate(tokenized_reviews):
        # Get the indices of the words in the review
        indices = [vocab.get(word, 0) for word in review]  # 0 for unknown words

        # Truncate or pad the sequence to the maximum length
        review_len = min(len(indices), max_len)
        review_indices[i, :review_len] = np.array(indices[:review_len], dtype=np.int32)

    return review_indices


# Custom Dataset Class (ClassificationDataset)
class ClassificationDataset(Dataset):
    """
    Sample Dataset for text classification tasks.

    Args:
        reviews (np.ndarray[str]): The reviews.
        labels (np.ndarray[np.int32]): The labels.
        vocab (dict[str, int]): The vocabulary dictionary, where the keys are the words and the values are their corresponding indices.
        max_len (int, optional): The maximum length of the sequence. Defaults to 100.
    """

    def __init__(self, reviews, labels, vocab, max_len=100):
        tokenized_reviews = tokenize_reviews(reviews)
        self.reviews_indices = reviews_to_indices(tokenized_reviews, vocab, max_len)
        self.labels = np.array(labels, dtype=np.int32)

    def __len__(self):
        return len(self.reviews_indices)

    def __getitem__(self, idx):
        return torch.tensor(self.reviews_indices[idx], dtype=torch.long), torch.tensor(
            self.labels[idx], dtype=torch.long
        )


# Step 4: Create DataLoader
def create_data_loader(
    data_dir: str, vocab: dict[str, int], batch_size: int = 32, shuffle: bool = True
) -> DataLoader[ClassificationDataset]:
    """
    Create a DataLoader from the given data directory and vocabulary.

    Args:
        data_dir (str): The path to the data directory.
        vocab (dict[str, int]): The vocabulary dictionary, where the keys are the words and the values are their corresponding indices.
        batch_size (int, optional): The batch size of the DataLoader. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the DataLoader. Defaults to True.

    Returns:
        DataLoader[ClassificationDataset]: The DataLoader.
    """
    reviews, labels = load_reviews(data_dir)
    dataset = ClassificationDataset(reviews, labels, vocab, SEQ_LENGTH)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


# CNN Block class
class CNNBlock(nn.Module):
    """
    A CNN block used for text classification tasks.

    Each block consists of a convolutional layer, a batch normalization layer, a ReLU activation, a max pooling layer, and a dropout layer.

    Attributes:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The size of the convolutional kernel.
        dropout (float, optional): The dropout probability. Defaults to 0.3.
        pool_size (int, optional): The size of the max pooling kernel. Defaults to 2.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, dropout=0.3, pool_size=2
    ):
        super(CNNBlock, self).__init__()
        # Convolutional Layer
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

        # Batch Normalization Layer
        self.batch_norm = nn.BatchNorm1d(out_channels)

        # Activation
        self.relu = nn.ReLU()

        # Max Pooling Layer
        self.pool = nn.MaxPool1d(pool_size)

        # Dropout Layer
        self.dropout = nn.Dropout(dropout)

        # Weight Initialization
        init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through one CNN block.

        Args:
        x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
        torch.Tensor: Output tensor of shape (batch_size, sequence_length, out_channels).
        """
        # Convolution + BatchNorm + ReLU
        x = self.relu(self.batch_norm(self.conv(x)))

        # Max Pooling
        x = self.pool(x)

        # Apply Dropout
        x = self.dropout(x)

        return x


# Main CNN Text Classifier class
class CNNTextClassifier(nn.Module):
    """
    A PyTorch module for text classification using convolutional neural networks (CNNs).

    Args:

    """

    def load_pretrained_embeddings(
        self, embedding_file: str, vocab: dict[str, int], embedding_dim: int = 200
    ) -> np.ndarray:
        """
        Load pre-trained embeddings from a file and update the embedding matrix.

        Args:
            embedding_file (str): The path to the pre-trained embeddings file.
            vocab (dict[str, int]): The vocabulary dictionary.
            embedding_dim (int, optional): The dimensionality of the embeddings. Defaults to 200.

        Returns:
            np.ndarray: The embedding matrix of shape (len(vocab) + 1, embedding_dim).
        """
        # Initialize the embedding matrix with random values
        embedding_matrix = np.random.uniform(
            -0.25, 0.25, (len(vocab) + 1, embedding_dim)
        )

        # Read the pre-trained embeddings file
        with open(embedding_file, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype="float32")
                # If the word is in our vocabulary, update its embedding
                if word in vocab:
                    idx = vocab[word]
                    embedding_matrix[idx] = vector

        return embedding_matrix

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        num_filters,
        filter_size,
        fc_dim,
        num_blocks=2,
        output_dim=1,
        dropout=0.3,
        sequence_length=None,
        use_pretrained_embedding=False,
        embedding_path=EMBEDDING_PATH,
        freeze_embedding=False,
        vocab=VOCAB,
    ):
        super(CNNTextClassifier, self).__init__()

        # Embedding Layer
        if (
            use_pretrained_embedding
            and embedding_path is not None
            and vocab is not None
        ):
            print(f"Loading pre-trained embeddings from {embedding_path}...")
            embedding_matrix = self.load_pretrained_embeddings(
                embedding_path, vocab, embedding_dim
            )
            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(embedding_matrix, dtype=torch.float),
                freeze=freeze_embedding,
            )
        else:
            # If no pre-trained embedding is provided, learn from scratch
            print("Learning embeddings from scratch...")
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # CNN Blocks
        self.cnn_blocks = nn.ModuleList()
        in_channels = embedding_dim

        # Create multiple CNN blocks
        for _ in range(num_blocks):
            block = CNNBlock(in_channels, num_filters, filter_size, dropout)
            self.cnn_blocks.append(block)
            in_channels = (
                num_filters  # The output of the previous block is the input to the next
            )

        # Fully connected layer size dynamically calculated
        self.fc = None  # Will initialize later dynamically

        # Output Layer
        self.fc_dim = fc_dim
        self.output = nn.Linear(fc_dim, output_dim)

        # Activation
        self.relu = nn.ReLU()

        self._initialize_weights()

    def forward(self, x):
        # x is of shape [batch_size, sequence_length]

        # Embedding lookup
        embedded = self.embedding(x)  # [batch_size, sequence_length, embedding_dim]

        # Permute for Conv1d (expects [batch_size, embedding_dim, sequence_length])
        embedded = embedded.permute(0, 2, 1)

        # Pass through each CNN block
        for block in self.cnn_blocks:
            embedded = block(embedded)

        # Flatten the output from the CNN blocks
        flat = embedded.view(
            embedded.size(0), -1
        )  # [batch_size, num_filters * final_pooled_sequence_length]

        # Dynamically calculate the input to the fully connected layer
        if self.fc is None:
            self.fc = nn.Linear(flat.size(1), self.fc_dim).to(
                flat.device
            )  # Initialize the fully connected layer dynamically and send to same device

        # Fully connected layer
        fc_out = self.fc(flat)

        # Output layer (returns a logit)
        output = self.output(fc_out)  # [batch_size, 1]

        return output.squeeze(1)  # Squeeze to get shape [batch_size]

    def _initialize_weights(self):
        # Initialize weights in fully connected layers
        if self.fc is not None:
            init.xavier_normal_(self.fc.weight)

        # Initialize the output layer
        init.xavier_normal_(self.output.weight)

        # Optionally, initialize biases to zero
        if self.fc is not None and self.fc.bias is not None:
            init.zeros_(self.fc.bias)
        init.zeros_(self.output.bias)


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    epochs=10,
):
    """
    Train a PyTorch model.

    Args:
        ...

    Returns:
        Lists of training and validation losses and accuracies.
    """
    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    start_time = time.time()  # Start timing

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        # Training loop
        for batch in train_loader:
            inputs, labels = batch

            # Send inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device).float()  # Convert labels to float

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (
                torch.sigmoid(outputs) > 0.5
            ).float()  # Thresholding the logits to get binary predictions
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        # Average training loss and accuracy
        train_loss /= len(train_loader)
        train_acc = correct_train / total_train

        # Append training metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation loop
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch

                # Send inputs and labels to the device
                inputs = inputs.to(device)
                labels = labels.to(device).float()  # Convert labels to float

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predicted = (
                    torch.sigmoid(outputs) > 0.5
                ).float()  # Thresholding for binary predictions
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        # Average validation loss and accuracy
        val_loss /= len(val_loader)
        val_acc = correct_val / total_val

        # Append validation metrics
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Update learning rate scheduler
        scheduler.step()

        # Print Epoch Summary
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    end_time = time.time()  # End timing
    training_time = end_time - start_time

    print(f"Time: {training_time:.2f}s")

    # Return the lists for plotting
    return train_losses, val_losses, train_accuracies, val_accuracies


# Custom configuration for WandB
config = {
    "batch_size": 32,
    "epochs": 20,
    "learning_rate": 0.001,
    "embedding_dim": 100,
    "num_filters": 70,
    "filter_size": 3,
    "fc_dim": 50,
    "num_blocks": 4,
    "output_dim": 1,
    "seq_length": SEQ_LENGTH,
    "model_name": "CNN-Text-Classifier",
}


# First, load the training data and build the vocabulary
train_reviews, train_labels = load_reviews(TRAIN_DIR)
tokenized_train_reviews = tokenize_reviews(train_reviews)
VOCAB = build_vocabulary(tokenized_train_reviews)


# Then, create the data loaders
train_loader = create_data_loader(TRAIN_DIR, VOCAB, batch_size=config["batch_size"])
test_loader = create_data_loader(
    TRAIN_DIR, VOCAB, batch_size=config["batch_size"], shuffle=False
)

# Example loop to visualize data
for review_batch, label_batch in train_loader:
    print(
        "Batch shape for reviews:", review_batch.shape
    )  # Expected shape: [batch_size, max_len]
    print("Batch shape for labels:", label_batch.shape)  # Expected shape: [batch_size]

    first_review = review_batch[0]
    first_label = label_batch[0]

    print("First review in the batch (shape):", first_review.shape)  # Shape: [max_len]
    print("First review (content):", first_review)  # Actual tensor content
    print("First label:", first_label)  # Label of the first review

    break


# Set hyperparameters based on the config
VOCAB_SIZE = len(VOCAB) + 1
EMBEDDING_DIM = config["embedding_dim"]
NUM_FILTERS = config["num_filters"]
FILTER_SIZE = config["filter_size"]
FC_DIM = config["fc_dim"]
NUM_BLOCKS = config["num_blocks"]
OUTPUT_DIM = config["output_dim"]

# Initialize the model
cnn_model = CNNTextClassifier(
    VOCAB_SIZE, EMBEDDING_DIM, NUM_FILTERS, FILTER_SIZE, FC_DIM, NUM_BLOCKS, OUTPUT_DIM
).to(device)

# Define loss function and optimizer
criterion = (
    nn.BCEWithLogitsLoss()
)  # Binary Cross Entropy Loss for binary classification
optimizer = optim.AdamW(
    cnn_model.parameters(), lr=config["learning_rate"], weight_decay=1e-4
)

# Define the Cosine Annealing Scheduler
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=0)


# Train the CNN model
train_losses, val_losses, train_accs, val_accs = train_model(
    cnn_model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    epochs=config["epochs"],
)


# Plot loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Time")
plt.legend()
plt.savefig("CNN_Training_vs_Testing_loss.png")

# Plot accuracy curves
plt.figure(figsize=(10, 5))
plt.plot(train_accs, label="Training Accuracy")
plt.plot(val_accs, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy over Time")
plt.legend()
plt.savefig("CNN_Training_vs_Testing_accuracy.png")


# Save the model
torch.save(cnn_model.state_dict(), "cnn_model.pt")


# Initialize the model with embeddings
cnn_model_with_embeddings = CNNTextClassifier(
    VOCAB_SIZE,
    EMBEDDING_DIM,
    NUM_FILTERS,
    FILTER_SIZE,
    FC_DIM,
    NUM_BLOCKS,
    OUTPUT_DIM,
    use_pretrained_embedding=True,
    freeze_embedding=True,
    embedding_path=EMBEDDING_PATH,
    vocab=VOCAB,
).to(device)

# Define loss function and optimizer
criterion = (
    nn.BCEWithLogitsLoss()
)  # Binary Cross Entropy Loss for binary classification
optimizer = optim.AdamW(
    cnn_model_with_embeddings.parameters(),
    lr=config["learning_rate"],
    weight_decay=1e-4,
)

# Define the Cosine Annealing Scheduler
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=0)

# Train the CNN model
train_losses, val_losses, train_accs, val_accs = train_model(
    cnn_model_with_embeddings,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    epochs=config["epochs"],
)


# Plot loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Time")
plt.legend()
plt.savefig("CNN_Embeddings_Training_vs_Testing_loss.png")

# Plot accuracy curves
plt.figure(figsize=(10, 5))
plt.plot(train_accs, label="Training Accuracy")
plt.plot(val_accs, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy over Time")
plt.legend()
plt.savefig("CNN_Embeddings_Training_vs_Testing_accuracy.png")


# Save the model
torch.save(cnn_model_with_embeddings.state_dict(), "cnn_embeddings_model.pt")
