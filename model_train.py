import torch
import torchvision
from torch import nn
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np

from model import CNN
import Resnet
import Alexnet
import VGG
import movie_dataset as md

# ===========================================================
# Global variables
# ===========================================================
EPOCH = 50  # number of times for each run-through
BATCH_SIZE = 64  # number of images for each epoch
LR = 0.001
GPU_IN_USE = True  # whether using GPU


# ===========================================================
# Prepare train dataset & test dataset
# ===========================================================
print("***** prepare data ******")
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])
training_set = md.MovieDataSet("train_data/train2.csv", training=True, transform=transform)
train_dataloader = torch.utils.data.DataLoader(dataset=training_set, batch_size=BATCH_SIZE, shuffle=True)

validation_set = md.MovieDataSet("train_data/validation2.csv", training=True, transform=transform)
validation_dataloader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=BATCH_SIZE, shuffle=True)
print("Finished")


# ===========================================================
# Prepare model
# ===========================================================
print("***** prepare model ******")
model = Resnet.resnet18(num_classes=4)
# model = torchvision.models.resnet50(num_classes=4)
# model = CNN()
# model = Alexnet.alexnet(num_classes=4)

if GPU_IN_USE:
    model.cuda()
# print(torch.cuda.device_count())
# print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss().cuda() if GPU_IN_USE else nn.CrossEntropyLoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
print("Finished")


# ===========================================================
# Train
# ===========================================================
def train(start_from=0, load_state=None):
    print("***** Train ******")
    accuracy_list = []
    loss_list = []

    if load_state is not None:
        model.load_state_dict(torch.load(load_state))

    for epoch in range(start_from, EPOCH):
        # Training
        train_acc_total = 0
        for iteration, ((x, info), y) in enumerate(train_dataloader):
            model.train()
            x = Variable(x).cuda() if GPU_IN_USE else Variable(x)
            info = Variable(info).cuda() if GPU_IN_USE else Variable(info)
            y = Variable(y).cuda() if GPU_IN_USE else Variable(y)

            output = model(x, info)
            loss = loss_func(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output.data, 1)
            train_accuracy = (predicted == y.data).sum().item() / y.data.shape[0]
            train_acc_total += train_accuracy
            print('Epoch: ', epoch, '| Iteration: ', iteration, '| train loss: %.4f' % loss.data,
                  '| train accuracy: %.2f' % train_accuracy)

        # Validation
        total = 0
        valid_correct = 0
        total_loss = 0
        model.eval()

        with torch.no_grad():
            for _, ((x_t, info_t), y_t) in enumerate(validation_dataloader):
                x_t = Variable(x_t).cuda() if GPU_IN_USE else Variable(x_t)
                info_t = Variable(info_t).cuda() if GPU_IN_USE else Variable(info_t)
                y_t = Variable(y_t).cuda() if GPU_IN_USE else Variable(y_t)

                output = model(x_t, info_t)
                _, predicted = torch.max(output.data, 1)
                loss = loss_func(output, y_t)
                total_loss += loss
                valid_correct += (predicted == y_t.data).sum().item()
                total += y_t.data.shape[0]

            valid_accuracy = float(valid_correct)/total
            accuracy_list.append(valid_accuracy)
            loss_list.append(total_loss)
            torch.save(model.state_dict(), "model_state/state18v_ep{}".format(epoch))
            print("Epoch:  {} -- Train accuracy: {}".format(epoch, train_acc_total / float(iteration + 1)))
            print("Epoch:  {} -- Validation accuracy: {}".format(epoch, valid_accuracy))
            print("*" * 50)

        if load_state is None or True:
            np.savetxt("record/accuracy_18v2.txt", np.array(accuracy_list))
            np.savetxt("record/loss_18v2.txt", np.array(loss_list))

# train(load_state="model_state/state_ep29", start_from=30)
# train(load_state="model_state/stateEX_ep29", start_from=30)
# train(load_state="model_state/stateEX2_ep29", start_from=30)
# train(load_state="model_state/state34_ep49", start_from=50)
# train(load_state="model_state/state34s_ep49", start_from=50)
train(load_state="model_state/state18v_ep29", start_from=30)
