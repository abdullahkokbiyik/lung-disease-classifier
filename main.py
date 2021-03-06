from XRAYDataset import XRAYDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch
import torch.optim as optim
from CNN import CNN
import time
import copy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN()
    #♦ lr=0.01 ile dene.
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()
    model = model.to(device)
    criterion = criterion.to(device)
    
    transform_train = transforms.Compose([
          transforms.Resize((224, 224)),
          transforms.RandomHorizontalFlip(),
          transforms.RandomRotation(12),
          transforms.ToTensor(),
          transforms.Normalize((0.5,), (0.5))
        ])
    
    """transform_test = transforms.Compose([
          transforms.Resize((224, 224)),
          transforms.ToTensor(),
          transforms.Normalize((0.5,), (0.5,))
        ])"""
    
    train_data = XRAYDataset('dataset/train/', transform=transform_train)
    train_val_dataset, test_dataset = train_test_split(train_data, test_size=0.2, shuffle=True)
    train_dataset, val_dataset = train_test_split(train_val_dataset, test_size=0.2, shuffle=True)
    # val_dataset = XRAYDataset('dataset/val/', transform=transform)
    # test_dataset = XRAYDataset('dataset/test/', transform=transform_test)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)
    
    train_val_loader = dict()
    train_val_loader['train'] = train_loader
    train_val_loader['val'] = val_loader
    
    best_model = train(model, train_val_loader, optimizer, criterion, device, 100)
    test(best_model, criterion, test_loader, device)


def train(model, dataloaders, optimizer, criterion, device, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 1.0
    loss_dict = dict()
    loss_dict['train'] = list()
    loss_dict['val'] = list()
    acc_dict = dict()
    acc_dict['train'] = list()
    acc_dict['val'] = list()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
    
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for data in dataloaders[phase]:
                inputs = data['image'].to(device)
                labels = data['category'].to(device)
    
    
                # zero the parameter gradients
                optimizer.zero_grad()
    
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
    
                    # backward + optimize only if in training phase
                    if phase == 'train':    
                        loss.backward()
                        optimizer.step()
    
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
    
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
          
            loss_dict[phase].append(epoch_loss)
            acc_dict[phase].append(epoch_acc)
    
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
    
        print()

    time_elapsed = time.time() - since
    plot("Loss", loss_dict['train'], loss_dict['val'], num_epochs)
    plot("Accuracy", acc_dict['train'], acc_dict['val'], num_epochs)
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val Loss: {:4f}'.format(best_loss))
    # load best model weights
    # Dataseti düzelttikten sonra burayı açacağız
    model.load_state_dict(best_model_wts)
    return model


def test(model, criterion, test_loader, device):

    test_loss = 0.
    correct = 0
    pred_list = list()
    correct_list = list()
    model.eval()

    for data in test_loader:
        image, category = data['image'].to(device), data['category'].to(device)
        target = model(image)
        loss = criterion(target, category)
        test_loss += loss.item() * image.size(0)
        _, pred = torch.max(target, 1)
        temp_pred = [a.item() for a in pred]
        temp_cat = [a.item() for a in category]
        pred_list.extend(temp_pred)
        correct_list.extend(temp_cat)
        correct += torch.sum(pred == category.data)
    
    epoch_loss = test_loss/len(test_loader.dataset)
    epoch_acc = correct.double()/len(test_loader.dataset)
    print('Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    return epoch_acc, epoch_loss, pred_list, correct_list


def plot(loss_or_acc, train, val, num_epoch):
    x = [i+1 for i in range(num_epoch)]
    plt.plot(x, train)
    plt.plot(x, val)
    plt.ylabel(loss_or_acc)
    plt.xlabel("Epochs")
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()


if __name__ == "__main__":
    main()
