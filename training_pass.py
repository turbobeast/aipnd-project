import torch
from torch import nn

def train_model(training_data, model, optimizer, epochs):
    criterion = nn.NLLLoss()
    train_losses = []

    for epoch in range(epochs):
        running_loss = 0
        accuracy = 0

        for iteration, (images, labels) in enumerate(training_data):
            print(iteration)
            if (iteration == 2):
              return
            # cuda_images = images.cuda()
            # cuda_labels = labels.cuda()
            optimizer.zero_grad()

            predictions = model(images)

            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            probabilities = torch.exp(predictions)
            _top_p, top_class = probabilities.topk(1, dim=1)
            equals = top_class.view(*labels.shape) == labels
            accuracy += torch.mean(equals.type(torch.FloatTensor))


        train_losses.append(running_loss/len(training_data))
        print("Accuracy: {}".format(accuracy/len(training_data)))
        print("Epoch: {}".format(epoch + 1))