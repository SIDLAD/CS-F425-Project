RunningInColab = False

# Imports
from torch.utils.data import Dataset, DataLoader
import os
import torch
import torchaudio
import torchaudio.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchsummary import summary

from math import ceil
from tqdm import tqdm

import matplotlib.pyplot as plt

# Hyperparameters
TARGET_SAMPLE_RATE = 16000
TARGET_LENGTH_SECONDS = 4
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 3e-4
DROPOUT_PROB = 0.4
DROPOUT_PROB_2D = 0.2

NUM_SAMPLES = TARGET_LENGTH_SECONDS * TARGET_SAMPLE_RATE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transformations = [
    transforms.MelSpectrogram(
            sample_rate=TARGET_SAMPLE_RATE,
            n_fft = 1024,
            hop_length = 512,
            n_mels = 64
        ),
]

class_mapping = {
  "car_horn":1,
  "dog_barking":2,
  "drilling":3,
  "Fart":4,
  "Guitar":5,
  "Gunshot_and_gunfire":6,
  "Hi-hat":7,
  "Knock":8,
  "Laughter":9,
  "Shatter":10,
  "siren":11,
  "Snare_drum":12,
  "Splash_and_splatter":13,
}

class AudioDataset(Dataset):
  def __init__(self,
               data_dir,
               transformations = transformations,
               target_sample_rate = TARGET_SAMPLE_RATE,
               num_samples = NUM_SAMPLES,
               device = device,
               testing = False
               ):
    self.data_dir = data_dir
    self.classes =[""]
    if(not testing):self.classes = sorted(os.listdir(data_dir))
    self.file_paths = []
    self.targets = []
    self.transformations = transformations
    self.target_sample_rate = target_sample_rate
    self.num_samples = num_samples
    self.device = device

    createMapping = False
    if(len(class_mapping) == 0):createMapping = True

    for i,class_name in enumerate(self.classes):
      class_dir = os.path.join(data_dir,class_name)
      if(createMapping):class_mapping[class_name] = i

      for filename in os.listdir(class_dir):
        filepath = os.path.join(class_dir,filename)
        self.file_paths.append(filepath)
        if(not testing): self.targets.append(class_mapping.get(class_name))
        else: self.targets.append(0)

  def __len__(self):
    return len(self.file_paths)

  def __getitem__(self,idx):
    audio_path = self.file_paths[idx]
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.to(self.device)
    waveform = self._resample_if_necessary(waveform,sample_rate)
    waveform = self._mix_down_if_necessary(waveform)
    waveform = self._cut_if_necessary(waveform)
    waveform = self._right_pad_if_necessary(waveform)


    if self.transformations:
      for transformation in self.transformations:
        waveform = transformation.to(self.device)(waveform)

    # waveform normalisation:
    waveform = torch.log1p(waveform)
    waveform = waveform * 255/(waveform.max() -waveform.min())

    label = self.targets[idx]
    return waveform, label

  def _resample_if_necessary(self, waveform,sample_rate):
    if sample_rate != self.target_sample_rate:
      resampler = torchaudio.transforms.Resample(sample_rate,self.target_sample_rate)
      waveform = resampler.to(self.device)(waveform)
    return waveform

  def _mix_down_if_necessary(self,waveform):
    if waveform.shape[0] > 1:
      waveform = torch.mean(waveform,dim = 0,keepdim = True)
    return waveform

  #If the video was longer than TARGET_LENGTH_SECONDS, then we are cropping it to that many seconds by removing seconds equally from both the start and the end sides
  def _cut_if_necessary(self,waveform):
    if waveform.shape[1] > self.num_samples:
      mid = (waveform.shape[1] - 1)//2
      nsby2 = self.num_samples//2
      waveform = waveform[:,mid - nsby2 + 1: mid + self.num_samples - nsby2 + 1]
    return waveform

  def _right_pad_if_necessary(self,waveform):
    num_samples = waveform.shape[1]
    if num_samples < self.num_samples:
      num_missing_samples = self.num_samples - num_samples
      last_dim_padding = (0,num_missing_samples)
      waveform = torch.nn.functional.pad(waveform,last_dim_padding)
    return waveform

class CustomResNet18(nn.Module):
    def __init__(self):
        super(CustomResNet18, self).__init__()
        self.resnet = models.resnet18(weights=None).to(device)

        self.resnet.conv1 = nn.Conv2d(1,64,kernel_size = 7,stride = 2,padding = 3,bias = False).to(device)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features,2048).to(device)
        # self.replace_batchnorm_layers(self.resnet)
        self.num_classes = len(class_mapping)

        self.resnet.conv1 = nn.Sequential(
            self.resnet.conv1,
            nn.Dropout2d(DROPOUT_PROB_2D)  # Add dropout after the first convolutional layer
        )
        # Add dropout after layers in resnet
        self.resnet.layer1 = nn.Sequential(
            self.resnet.layer1,
            nn.Dropout2d(DROPOUT_PROB_2D)  # Add dropout after layer1
        )
        self.resnet.layer2 = nn.Sequential(
            self.resnet.layer2,
            nn.Dropout2d(DROPOUT_PROB_2D)  # Add dropout after layer2
        )
        self.resnet.layer3 = nn.Sequential(
            self.resnet.layer3,
            nn.Dropout2d(DROPOUT_PROB_2D)  # Add dropout after layer3
        )
        self.resnet.layer4 = nn.Sequential(
            self.resnet.layer4,
            nn.Dropout2d(DROPOUT_PROB_2D)  # Add dropout after layer4
        )

        # Add additional linear layers
        self.additional_layers = nn.ModuleList().to(device)
        self.additional_layers.append(nn.Dropout(DROPOUT_PROB).to(device))
        self.additional_layers.append(nn.Linear(2048, 1024).to(device))
        self.additional_layers.append(nn.Linear(1024,512).to(device))
        self.additional_layers.append(nn.ReLU().to(device))
        self.additional_layers.append(nn.Linear(512,self.num_classes).to(device))

    def forward(self, x):
        x = self.resnet(x)
        for layer in self.additional_layers:
            x = layer(x)
        return x

if __name__ == "__main__":
  # Define data directories
  train_dir: str
  val_dir:str
  if RunningInColab:
      train_dir = "/content/drive/MyDrive/audio_dataset/train"
      val_dir = "/content/drive/MyDrive/audio_dataset/val"
  else:
      train_dir = "audio_dataset/train"
      val_dir = "audio_dataset/val"

  train_dataset = AudioDataset(train_dir)
  val_dataset = AudioDataset(val_dir)

  train_loader = DataLoader(
      train_dataset,batch_size = BATCH_SIZE, shuffle = True
  )
  val_loader = DataLoader(
      val_dataset,batch_size = BATCH_SIZE
  )

  model = CustomResNet18()
  summary(model,(1,64,126))

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(),lr = LEARNING_RATE)


  def train_model(model, train_loader, criterion, optimizer, epoch):
    model.train()  # Set model to training mode
    running_loss = 0.0

    correct = 0
    total = len(train_dataset)

    num_classes = len(class_mapping)
    classwise_true_positive = torch.tensor([0]*num_classes).to(device)
    classwise_true_negative = torch.tensor([0]*num_classes).to(device)
    classwise_false_positive = torch.tensor([0]*num_classes).to(device)
    classwise_false_negative = torch.tensor([0]*num_classes).to(device)

    batch_itr = 0
    loop = tqdm(train_loader, leave=True, desc=f"Training Epoch {epoch+1}/{EPOCHS}",bar_format="{l_bar}{bar}|")

    for inputs, labels in train_loader:
      batch_itr+=1

      inputs = inputs.to(device)
      labels = labels.to(device) - torch.ones_like(labels).to(device)

      optimizer.zero_grad()

      # Forward pass
      outputs = model(inputs)
      _, predicted = torch.max(outputs, 1)

      loss = criterion(outputs, labels)

      correct += (predicted == labels).sum().item()

      for _,class_number in class_mapping.items():
        classwise_predicted = (predicted == (class_number-1))
        classwise_labels = (labels == (class_number-1))
        classwise_true_positive[class_number-1] += torch.logical_and((classwise_predicted == classwise_labels),(classwise_predicted == 1)).sum().item()
        classwise_true_negative[class_number-1] += torch.logical_and((classwise_predicted == classwise_labels),(classwise_predicted == 0)).sum().item()
        classwise_false_positive[class_number-1] += torch.logical_and((classwise_predicted != classwise_labels),(classwise_predicted == 1)).sum().item()
        classwise_false_negative[class_number-1] += torch.logical_and((classwise_predicted != classwise_labels),(classwise_predicted == 0)).sum().item()


      # Backward pass and optimize
      loss.backward()
      optimizer.step()

      running_loss += loss.item() * inputs.size(0)
      loop.update()

    loop.close()

    # will have to print classwise precision accuracy and recall as well here

    epoch_loss = running_loss / len(train_loader.dataset)

    classwise_train_accuracy = (classwise_true_positive + classwise_true_negative)/(classwise_true_positive + classwise_true_negative + classwise_false_positive + classwise_false_negative)
    classwise_train_precision = (classwise_true_positive)/(classwise_true_positive + classwise_false_positive)
    classwise_train_recall = (classwise_true_positive)/(classwise_true_positive + classwise_false_negative)

    train_accuracy = correct / total
    train_precision = torch.mean(classwise_train_precision).item()
    train_recall = torch.mean(classwise_train_recall).item()

    classwise_train_accuracy = [round(element, 4) for element in classwise_train_accuracy.tolist()]
    classwise_train_precision = [round(element, 4) for element in classwise_train_precision.tolist()]
    classwise_train_recall = [round(element, 4) for element in classwise_train_recall.tolist()]

    print(f"Epoch [{epoch+1}/{EPOCHS}], Training Loss: {epoch_loss:.4f}")
    print(f'Training Accuracy: {train_accuracy:.4f}')
    print(f'Training Precision: {train_precision:.4f}')
    print(f'Training Recall: {train_recall:.4f}')
    print(f'Classwise Training Accuracy: {classwise_train_accuracy}')
    print(f'Classwise Training Precision: {classwise_train_precision}')
    print(f'Classwise Training Recall: {classwise_train_recall}\n')

    return train_accuracy,epoch_loss,train_precision,train_recall

  # Validation function
  def validate_model(model, val_loader):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0

    correct = 0
    total = len(val_dataset)

    num_classes = len(class_mapping)
    classwise_true_positive = torch.tensor([0]*num_classes).to(device)
    classwise_true_negative = torch.tensor([0]*num_classes).to(device)
    classwise_false_positive = torch.tensor([0]*num_classes).to(device)
    classwise_false_negative = torch.tensor([0]*num_classes).to(device)

    batch_itr = 0
    loop = tqdm(val_loader, leave=True, desc=f"Validation:",bar_format="{l_bar}{bar}|")

    with torch.no_grad():
      for inputs, labels in val_loader:
        batch_itr+=1

        inputs = inputs.to(device)
        labels = labels.to(device) - torch.ones_like(labels).to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        loss = criterion(outputs, labels)

        correct += (predicted == labels).sum().item()

        for _,class_number in class_mapping.items():
          classwise_predicted = (predicted == (class_number-1))
          classwise_labels = (labels == (class_number-1))
          classwise_true_positive[class_number-1] += torch.logical_and((classwise_predicted == classwise_labels),(classwise_predicted == 1)).sum().item()
          classwise_true_negative[class_number-1] += torch.logical_and((classwise_predicted == classwise_labels),(classwise_predicted == 0)).sum().item()
          classwise_false_positive[class_number-1] += torch.logical_and((classwise_predicted != classwise_labels),(classwise_predicted == 1)).sum().item()
          classwise_false_negative[class_number-1] += torch.logical_and((classwise_predicted != classwise_labels),(classwise_predicted == 0)).sum().item()



        running_loss += loss.item() * inputs.size(0)
        loop.update()

    loop.close()

    # will have to print classwise precision accuracy and recall as well here

    epoch_loss = running_loss / len(val_loader.dataset)
    print(f"Validation Loss: {epoch_loss:.4f}")

    classwise_val_accuracy = (classwise_true_positive + classwise_true_negative)/(classwise_true_positive + classwise_true_negative + classwise_false_positive + classwise_false_negative)
    classwise_val_precision = (classwise_true_positive)/(classwise_true_positive + classwise_false_positive)
    classwise_val_recall = (classwise_true_positive)/(classwise_true_positive + classwise_false_negative)

    val_accuracy = correct / total
    val_precision = torch.mean(classwise_val_precision).item()
    val_recall = torch.mean(classwise_val_recall).item()

    classwise_val_accuracy = [round(element, 4) for element in classwise_val_accuracy.tolist()]
    classwise_val_precision = [round(element, 4) for element in classwise_val_precision.tolist()]
    classwise_val_recall = [round(element, 4) for element in classwise_val_recall.tolist()]

    print(f'Validation Accuracy: {val_accuracy:.4f}')
    print(f'Validation Precision: {val_precision:.4f}')
    print(f'Validation Recall: {val_recall:.4f}')
    print(f'Classwise Validation Accuracy: {classwise_val_accuracy}')
    print(f'Classwise Validation Precision: {classwise_val_precision}')
    print(f'Classwise Validation Recall: {classwise_val_recall}\n')

    return val_accuracy,epoch_loss,val_precision,val_recall

  # Training Loop
  best_accuracy = 0.0

  train_accuracy_list=[]
  train_loss_list=[]
  train_precision_list=[]
  train_recall_list=[]
  val_accuracy_list=[]
  val_loss_list=[]
  val_precision_list=[]
  val_recall_list=[]

  for epoch in range(EPOCHS):
    train_acc,train_loss,train_prec,train_rec = train_model(model,train_loader,criterion,optimizer,epoch)
    train_accuracy_list.append(train_acc)
    train_loss_list.append(train_loss)
    train_precision_list.append(train_prec)
    train_recall_list.append(train_rec)

    val_acc,val_loss,val_prec,val_rec = validate_model(model, val_loader)
    val_accuracy_list.append(val_acc)
    val_loss_list.append(val_loss)
    val_precision_list.append(val_prec)
    val_recall_list.append(val_rec)

    if((epoch+1) % 10 == 0 or (epoch+1) == EPOCHS):
      filename = f"epoch{epoch+1}.pth"
      torch.save(model.state_dict(),os.path.join(filename)) #This creates a .pth file every 10 epochs
    if val_acc > best_accuracy:best_accuracy = val_acc

    print(f"Best Validation Accuracy: {best_accuracy:.4f}\n")

  epochs = range(1, EPOCHS + 1)
  # Create subplots
  fig, axs = plt.subplots(2, 2, figsize=(10, 8))

  # Plot accuracy
  axs[0, 0].plot(epochs, train_accuracy_list, label = 'Training Accuracy', color='blue', marker='o')
  axs[0, 0].plot(epochs, val_accuracy_list, label='Validation Accuracy', color='orange', marker='o')
  axs[0, 0].set_title('Accuracy Comparison')
  axs[0, 0].set_xlabel('Epoch')
  axs[0, 0].set_ylabel('Accuracy')
  axs[0, 0].legend()

  # Plot loss
  axs[0, 1].plot(epochs, train_loss_list, label = 'Training Loss', color='blue', marker='o')
  axs[0, 1].plot(epochs, val_loss_list, label='Validation Loss', color='orange', marker='o')
  axs[0, 1].set_title('Loss Comparison')
  axs[0, 1].set_xlabel('Epoch')
  axs[0, 1].set_ylabel('Loss')
  axs[0, 1].legend()

  # Plot recall
  axs[1, 0].plot(epochs, train_recall_list, label = 'Training Recall', color='blue', marker='o')
  axs[1, 0].plot(epochs, val_recall_list, label= 'Validation Recall', color='orange', marker='o')
  axs[1, 0].set_title('Recall Comparison')
  axs[1, 0].set_xlabel('Epoch')
  axs[1, 0].set_ylabel('Recall')
  axs[1, 0].legend()

  # Plot precision
  axs[1, 1].plot(epochs, train_precision_list, label = 'Training Precision', color='blue', marker='o')
  axs[1, 1].plot(epochs, val_precision_list, label='Validaton Precision', color='orange', marker='o')
  axs[1, 1].set_title('Precision Comparison')
  axs[1, 1].set_xlabel('Epoch')
  axs[1, 1].set_ylabel('Precision')
  axs[1, 1].legend()

  # Adjust layout
  plt.tight_layout()

  # Show plot
  plt.show()