import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import json
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser
import sys
import time

# Unknown token
unk = '<UNK>'

class RNN(nn.Module):
    def __init__(self, input_dim, h, embeddings, dropout=0.5):
        super(RNN, self).__init__()
        self.h = h
        self.num_layers = 1

        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embeddings, dtype=torch.float), freeze=False)
        # Define the RNN layer
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=h, num_layers=self.num_layers, nonlinearity='tanh', batch_first=False)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        # Output layer
        self.fc = nn.Linear(h, 5)  # Assuming 5 sentiment classes (0 to 4)

        self.loss_fn = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss_fn(predicted_vector, gold_label)

    def forward(self, inputs):
        embedded_inputs = self.embedding(inputs)  # [seq_len, batch_size, embedding_dim]
        output, hidden = self.rnn(embedded_inputs)  # hidden: [num_layers, batch_size, hidden_size]
        last_hidden = hidden[-1]  # [batch_size, hidden_size]
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)  # [batch_size, num_classes]
        predicted_vector = nn.functional.log_softmax(logits, dim=1)
        return predicted_vector

def load_data(train_data, val_data, test_data):
    with open(train_data) as f:
        training = json.load(f)
    with open(val_data) as f:
        validation = json.load(f)
    with open(test_data) as f:
        testing = json.load(f)

    tra = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in training]
    val = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in validation]
    tst = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in testing]

    return tra, val, tst

def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        indices = [word2index.get(word, word2index[unk]) for word in document]
        vector = torch.tensor(indices, dtype=torch.long)
        vectorized_data.append((vector, y))
    return vectorized_data

if __name__ == "__main__":
    sys.argv = ['rnn.py', '--hidden_dim', '256', '--epochs', '10', '--train_data', 'training.json',
                '--val_data', 'validation.json', '--test_data', 'test.json', '--embedding_file', 'word_embedding.pkl', '--do_train']

    parser = ArgumentParser()
    parser.add_argument("--hidden_dim", type=int, required=True, help="Hidden dimension")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs to train")
    parser.add_argument("--train_data", required=True, help="Path to training data")
    parser.add_argument("--val_data", required=True, help="Path to validation data")
    parser.add_argument("--test_data", required=True, help="Path to test data")
    parser.add_argument("--embedding_file", required=True, help="Path to word_embedding.pkl")
    parser.add_argument("--do_train", action='store_true', help="Flag to indicate if training should be performed")
    parser.add_argument("--batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--save_path", default="model_checkpoint.pth", help="Path to save model checkpoints")
    args = parser.parse_args()

    # Fix random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    # Load data
    print("========== Loading data ==========")
    train_data, valid_data, test_data = load_data(args.train_data, args.val_data, args.test_data)

    print("========== Loading word embeddings and creating word2index ==========")

    with open(args.embedding_file, 'rb') as f:
        word_embedding_data = pickle.load(f)

    words_list = list(word_embedding_data.keys())
    embeddings = list(word_embedding_data.values())

    word2index = {word: idx for idx, word in enumerate(words_list)}

    if unk not in word2index:
        word2index[unk] = len(word2index)
        embedding_dim = len(embeddings[0])
        embeddings.append(np.random.uniform(-0.25, 0.25, embedding_dim))

    embeddings = np.array(embeddings)

    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)
    test_data = convert_to_vector_representation(test_data, word2index)

    model = RNN(input_dim=embeddings.shape[1], h=args.hidden_dim, embeddings=embeddings, dropout=0.5)

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    gradient_clip_value = 1.0
    patience = 3

    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []

    best_val_loss = float('inf')
    early_stop_counter = 0

    # 1) Train the model
    if args.do_train:
        print(f"========== Training for {args.epochs} epochs ==========")
        for epoch in range(args.epochs):
            model.train()
            correct = 0
            total = 0
            epoch_loss = 0
            start_time = time.time()

            random.shuffle(train_data)
            minibatch_size = args.batch_size
            N = len(train_data)

            for minibatch_index in tqdm(range((N + minibatch_size - 1) // minibatch_size)):
                optimizer.zero_grad()
                loss = 0
                batch_correct = 0
                batch_total = 0
                batch_start = minibatch_index * minibatch_size
                batch_end = min(batch_start + minibatch_size, N)
                for index in range(batch_start, batch_end):
                    input_vector, gold_label = train_data[index]
                    input_vector = input_vector.unsqueeze(1)
                    predicted_vector = model(input_vector)
                    predicted_label = torch.argmax(predicted_vector, dim=1).item()
                    batch_correct += int(predicted_label == gold_label)
                    batch_total += 1
                    example_loss = model.compute_Loss(predicted_vector, torch.tensor([gold_label]))
                    loss += example_loss
                loss = loss / batch_total
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)

                optimizer.step()
                epoch_loss += loss.item() * batch_total
                correct += batch_correct
                total += batch_total

            training_losses.append(epoch_loss / total)
            training_accuracies.append(correct / total)

            # Save model after each epoch
            torch.save(model.state_dict(), args.save_path)

            print(f"Training completed for epoch {epoch + 1}")
            print(f"Training accuracy for epoch {epoch + 1}: {correct / total:.4f}")
            print(f"Training loss for epoch {epoch + 1}: {epoch_loss / total:.4f}")

            # Validation loop
            model.eval()
            correct = 0
            total = 0
            epoch_loss = 0

            with torch.no_grad():
                minibatch_size = args.batch_size
                N_val = len(valid_data)

                for minibatch_index in tqdm(range((N_val + minibatch_size - 1) // minibatch_size)):
                    loss = 0
                    batch_correct = 0
                    batch_total = 0
                    batch_start = minibatch_index * minibatch_size
                    batch_end = min(batch_start + minibatch_size, N_val)
                    for index in range(batch_start, batch_end):
                        input_vector, gold_label = valid_data[index]
                        input_vector = input_vector.unsqueeze(1)
                        predicted_vector = model(input_vector)
                        predicted_label = torch.argmax(predicted_vector, dim=1).item()
                        batch_correct += int(predicted_label == gold_label)
                        batch_total += 1
                        example_loss = model.compute_Loss(predicted_vector, torch.tensor([gold_label]))
                        loss += example_loss
                    epoch_loss += loss.item()
                    correct += batch_correct
                    total += batch_total

            validation_losses.append(epoch_loss / total)
            validation_accuracies.append(correct / total)

            if epoch_loss / total < best_val_loss:
                best_val_loss = epoch_loss / total
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print("Early stopping triggered")
                break

    print("Training finished.")

    # 2) Test accuracy prediction
    with open("test.json") as f:
        test_data = json.load(f)

    test_samples = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in test_data]
    vectorized_test_data = convert_to_vector_representation(test_samples, word2index)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for input_vector, gold_label in vectorized_test_data:
            input_vector = input_vector.unsqueeze(1)
            predicted_vector = model(input_vector)
            predicted_label = torch.argmax(predicted_vector, dim=1).item()

            if predicted_label == gold_label:
                correct += 1
            total += 1

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

    # 3) Plotting training vs validation loss
    epochs = range(1, len(training_losses) + 1)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_losses, label='Training Loss', color='blue', marker='o')
    plt.plot(epochs, validation_losses, label='Validation Loss', color='orange', marker='x')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 4) Plotting training vs validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, training_accuracies, label='Training Accuracy', color='blue', marker='o')
    plt.plot(epochs, validation_accuracies, label='Validation Accuracy', color='orange', marker='x')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 5) Plotting training loss vs validation accuracy
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color='blue')
    ax1.plot(epochs, training_losses, label='Training Loss', color='blue', marker='o')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Accuracy', color='orange')
    ax2.plot(epochs, validation_accuracies, label='Validation Accuracy', color='orange', marker='x')
    ax2.tick_params(axis='y', labelcolor='orange')

    plt.title('Training Loss and Validation Accuracy vs. Epochs')
    fig.tight_layout()
    plt.grid()
    plt.show()
