import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import sys

unk = '<UNK>'

# Define the Feedforward Neural Network (FFNN) model
class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)  # Defined first fully connected/input layer
        self.activation = nn.ReLU()  # Apply ReLU activation
        self.dropout = nn.Dropout(0.5)  # Dropout to prevent overfitting
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)  # Second fully connected/output layer
        self.softmax = nn.LogSoftmax(dim=-1)  # Apply LogSoftmax
        self.loss = nn.NLLLoss()  # Negative Log Likelihood Loss

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        hidden_rep = self.activation(self.W1(input_vector))  # Hidden layer with ReLU activation
        hidden_rep = self.dropout(hidden_rep)  # Apply dropout
        output_logits = self.W2(hidden_rep)  # Output layer
        predicted_vector = self.softmax(output_logits)  # LogSoftmax
        return predicted_vector


# Function to create vocabulary from data
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab


# Function to create word to index and index to word mappings
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {word: index for index, word in enumerate(vocab_list)}
    index2word = {index: word for index, word in enumerate(vocab_list)}
    vocab.add(unk)
    return vocab, word2index, index2word


# Function to convert data into vector representations
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index))
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data


# Function to load training, validation, and testing data
def load_data(train_data, val_data, test_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)
    with open(test_data) as test_f:
        testing = json.load(test_f)

    tra = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in training]
    val = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in validation]
    tst = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in testing]

    return tra, val, tst


# Main execution starts here
if __name__ == "__main__":
    sys.argv = ['ffnn.py', '--hidden_dim', '256', '--epochs', '20', '--train_data', 'training.json',
                '--val_data', 'validation.json', '--test_data', 'test.json', '--do_train']

    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", required=True, help="path to test data")
    parser.add_argument("--do_train", action='store_true', help="Flag to indicate if training should be performed")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="minibatch size")
    args = parser.parse_args()

    # Fix random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    # Load data
    print("========== Loading data ==========")
    train_data, valid_data, test_data = load_data(args.train_data, args.val_data, args.test_data)
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)
    test_data = convert_to_vector_representation(test_data, word2index)

    # Initialize the model, optimizer, and training configurations
    model = FFNN(input_dim=len(vocab), h=args.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Added weight decay

    # Early stopping setup
    best_val_loss = float('inf')
    patience = 3
    early_stop_counter = 0

    # Training, Validation, and Early Stopping
    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []

    # Open the results file
    with open('results_ffnn.txt', 'w') as f:
        f.write("Epoch, Training Loss, Validation Loss, Training Accuracy, Validation Accuracy\n")

        if args.do_train:
            print("========== Training for {} epochs ==========".format(args.epochs))
            for epoch in range(args.epochs):
                model.train()
                correct = 0
                total = 0
                epoch_loss = 0
                start_time = time.time()
                print(f"Training started for epoch {epoch + 1}")

                random.shuffle(train_data)
                minibatch_size = args.batch_size
                N = len(train_data)

                # Training loop
                for minibatch_index in tqdm(range(N // minibatch_size)):
                    optimizer.zero_grad()
                    loss = 0
                    for example_index in range(minibatch_size):
                        input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                        input_vector = input_vector.float()

                        predicted_vector = model(input_vector)
                        predicted_label = torch.argmax(predicted_vector)

                        correct += int(predicted_label == gold_label)
                        total += 1
                        example_loss = model.compute_Loss(predicted_vector.view(1, -1), torch.tensor([gold_label]))
                        loss += example_loss
                    loss = loss / minibatch_size
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                # Track training accuracy and loss
                training_losses.append(epoch_loss / (N // minibatch_size))
                training_accuracies.append(correct / total)

                print(f"Training completed for epoch {epoch + 1}")
                print(f"Training accuracy for epoch {epoch + 1}: {correct / total}")
                print(f"Training loss for epoch {epoch + 1}: {epoch_loss / (N // minibatch_size)}")
                print(f"Training time for this epoch: {time.time() - start_time}")

                # Validation loop
                model.eval()
                correct = 0
                total = 0
                epoch_loss = 0
                start_time = time.time()
                print(f"Validation started for epoch {epoch + 1}")

                minibatch_size = 16
                N = len(valid_data)

                with torch.no_grad():
                    for minibatch_index in tqdm(range(N // minibatch_size)):
                        loss = 0
                        for example_index in range(minibatch_size):
                            input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                            input_vector = input_vector.float()

                            predicted_vector = model(input_vector)
                            predicted_label = torch.argmax(predicted_vector)

                            correct += int(predicted_label == gold_label)
                            total += 1
                            example_loss = model.compute_Loss(predicted_vector.view(1, -1), torch.tensor([gold_label]))
                            loss += example_loss
                        epoch_loss += loss.item()

                # Track validation accuracy and loss
                validation_losses.append(epoch_loss / (N // minibatch_size))
                validation_accuracies.append(correct / total)

                print(f"Validation completed for epoch {epoch + 1}")
                print(f"Validation accuracy for epoch {epoch + 1}: {correct / total}")
                print(f"Validation loss for epoch {epoch + 1}: {epoch_loss / (N // minibatch_size)}")
                print(f"Validation time for this epoch: {time.time() - start_time}")

                # Write results to file
                f.write(f"{epoch + 1}, {training_losses[-1]}, {validation_losses[-1]}, "
                        f"{training_accuracies[-1]}, {validation_accuracies[-1]}\n")

                # Early stopping based on validation loss
                if validation_losses[-1] < best_val_loss:
                    best_val_loss = validation_losses[-1]
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                if early_stop_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        # Testing loop: compute accuracy for test data
        print("========== Testing ==========")
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for input_vector, gold_label in test_data:
                input_vector = input_vector.float()

                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)

                correct += int(predicted_label == gold_label)
                total += 1
        test_accuracy = correct / total
        print(f"Test accuracy: {test_accuracy}")
        f.write(f"Test accuracy: {test_accuracy}\n")

    # Plot 1: Training Loss vs. Validation Loss
    epochs = range(1, len(training_losses) + 1)
    plt.figure(figsize=(12, 6))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_losses, label='Training Loss', color='blue', marker='o')
    plt.plot(epochs, validation_losses, label='Validation Loss', color='orange', marker='x')
    plt.title('Training Loss vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot 2: Training Accuracy vs. Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, training_accuracies, label='Training Accuracy', color='blue', marker='o')
    plt.plot(epochs, validation_accuracies, label='Validation Accuracy', color='orange', marker='x')
    plt.title('Training Accuracy vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Plot 3: Training Loss vs. Validation Accuracy
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color='blue')
    ax1.plot(epochs, training_losses, label='Training Loss', color='blue', marker='o')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Accuracy', color='orange')
    ax2.plot(epochs, validation_accuracies, label='Validation Accuracy', color='orange', marker='x')
    ax2.tick_params(axis='y', labelcolor='orange')

    plt.title('Training Loss vs Validation Accuracy')
    fig.tight_layout()
    plt.grid(True)
    plt.show()
