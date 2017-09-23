#pylint: disable=E1101,R,C
import torch
import torch.nn as nn
from torch.autograd import Variable
from architecture import Mnist_Classifier
from utils import load_data

# data
MNIST_PATH =  "s2_mnist.gz"

# which CUDA device (if available)
DEVICE_ID = 0

# hyper parameters
NUM_EPOCHS = 1000
BATCH_SIZE = 128
LEARNING_RATE = 1e-4

def main():

    # load data
    train_loader, test_loader, train_dataset, _ = load_data(
        MNIST_PATH, BATCH_SIZE)

    # set CUDA mode
    torch.cuda.set_device(DEVICE_ID)

    # init model
    classifer = Mnist_Classifier()

    # set CUDA mode for model
    if torch.cuda.is_available():
        classifer.cuda(DEVICE_ID)

    # use cross entropy loss for the digits
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda(DEVICE_ID)

    # use Adam as optimizer
    optimizer = torch.optim.Adam(
        classifer.parameters(),
        lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images)
            labels = Variable(labels)

            if torch.cuda.is_available():
                images = images.cuda(DEVICE_ID)
                labels = labels.cuda(DEVICE_ID)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = classifer(images)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            print('Epoch [{0}/{1}], Iter [{2}/{3}] Loss: {4:.4f}'.format(
                epoch+1, NUM_EPOCHS, i+1, len(train_dataset)//BATCH_SIZE,
                loss.data[0]))

    # Test the Model
    correct = 0
    total = 0

    if torch.cuda.is_available():
        classifer.cuda(DEVICE_ID)

    for i, (images, labels) in enumerate(test_loader):
        images = Variable(images, volatile=True)
        if torch.cuda.is_available():
            images = images.cuda(DEVICE_ID)
            labels = labels.cuda(DEVICE_ID)

        outputs = classifer(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Test Accuracy of the model '\
          'on the 10000 test images: {0}'.format(100 * correct / total))

if __name__ == '__main__':
    main()
