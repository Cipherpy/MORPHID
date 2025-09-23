# MORPHID
Morphological Feature Generation for Species Identification

## Baseline  

As a foundational benchmark, we developed finetuned **convolutional neural network (CNN) models** on a curated dataset comprising **30 taxonomically diverse fish otolith classes**.  
The dataset includes representatives such as:  

*Alepocephalus bicolor, Apistus carinatus, Brachypterois serrulata, Chlorophthalmus acutifrons, Choridactylus multibarbus, Coryphaenoides sp., Cubiceps baxteri, Dactyloptena orientalis, Dactyloptena papilio, Dactyloptena tiltoni, Ectreposbastes imus, Grammoplites suppositus, Hoplostethus sp., Lepidotrigla spiloptera, Minous dempsterae, Minous inermis, Minous trachycephalus, Neomerinthe erostris, Parascombrops pellucidus, Platycephalus indicus, Polymixia fusca, Psenopsis sp., Pterygotrigla arabica, Pterygotrigla hemisticta, Pterygotrigla macrorhynchus, Satyrichthys laticeps, Setarches guentheri, Sorsogona tuberculata, Synagrops japonicus,* and *Uranoscopus sp.*  

### Dataset Organization  
- The dataset was stratified into **training**, **validation**, and **test** partitions.  
- All data are systematically arranged in the [`/dataset/`](./cnn/dataset) directory.  

### Training Protocol  
Finetuning was performed using modular code provided in the [`cnn/`](./cnn/) framework.  
A typical training run can be executed as:  

```bash
python -m cnn.scripts.main --mode train --model resnet50 --data_dir ./cnn/dataset
```

### Hyperparameters  
Hyperparameters such as **learning rate**, **batch size**, **optimizer**, and **input resolution** are configurable via [`cnn/scripts/main.py`](./cnn/scripts/main.py).  

The pipeline supports multiple architectures (**ResNet, VGG, DenseNet, EfficientNet**, etc.) for comparative benchmarking.  

---

### Outputs and Reproducibility  

- Trained models are automatically stored in timestamped directories under  
  [`cnn/outputs/models`](./cnn/outputs/models).  

- Ancillary outputs are generated, including:  
  - Grad-CAM visualizations  
  - Metrics (accuracy, precision, recall, F1)  
  - Confusion matrices  
  - Loss, accuracy, precision, and recall curves across epochs  

All plots are saved in the [`plots`](./cnn/outputs/plots) subfolder within the corresponding timestamped output directory.  




