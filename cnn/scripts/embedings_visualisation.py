import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import resnet152
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import plotly.express as px
import pandas as pd
from sklearn.manifold import TSNE


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = resnet152(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 30)  # Assuming your trained model has 30 classes
model.load_state_dict(torch.load('/home/reshma/Otolith/otolith_Final/src/outputs/20250515_095917/models/resnet152_no_augment_color_size512_best.pth', map_location=device))
model.fc = nn.Identity()  # Remove the classification head for embeddings
model.to(device)
model.eval()

# Data transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

# Test data loading
test_dataset = ImageFolder('/home/reshma/Otolith/otolith_Final/model_data/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Extract embeddings
embeddings, labels = [], []

with torch.no_grad():
    for images, targets in tqdm(test_loader):
        images = images.to(device)
        feats = model(images)
        embeddings.append(feats.cpu())
        labels.extend(targets.cpu().numpy())

embeddings = torch.cat(embeddings).numpy()
labels = np.array(labels)



# Run t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d = tsne.fit_transform(embeddings)

# Class names
class_names = test_dataset.classes
num_classes = len(class_names)

# Plot embeddings
plt.figure(figsize=(12, 10))
palette = sns.color_palette("tab20", num_classes)

for i in range(num_classes):
    idx = labels == i
    plt.scatter(
        embeddings_2d[idx, 0], 
        embeddings_2d[idx, 1],
        label=class_names[i],
        color=palette[i],
        alpha=0.7,
        s=50
    )

plt.title('t-SNE Visualization of ResNet-152 Embeddings', fontsize=15)
plt.xlabel('t-SNE Dimension 1', fontsize=12)
plt.ylabel('t-SNE Dimension 2', fontsize=12)
plt.legend(loc='best', bbox_to_anchor=(1.02, 1), fontsize='small')
plt.grid(False)
plt.tight_layout()
plt.savefig('resnet152_tsne_test.png', dpi=1200)
plt.show()

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d = tsne.fit_transform(embeddings)

# Create DataFrame for Plotly
df = pd.DataFrame({
    'x': embeddings_2d[:, 0],
    'y': embeddings_2d[:, 1],
    'label': [test_dataset.classes[l] for l in labels]
})

# Interactive Plotly scatter plot
fig = px.scatter(df, x='x', y='y', color='label',
                 title="Interactive t-SNE Embedding Visualization",
                 labels={"x": "t-SNE Dim 1", "y": "t-SNE Dim 2"})

fig.update_traces(marker=dict(size=6, opacity=0.8))
fig.update_layout(legend_title_text='Species')
# --- save the composite figure ---


fig.show()