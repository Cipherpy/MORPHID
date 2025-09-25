# MORPHID
Morphological Feature Generation for Species Identification

## Baseline (Stage I: Foundational Benchmark)

As a foundational benchmark, we developed finetuned **convolutional neural network (CNN) models** on a curated dataset comprising **30 taxonomically diverse fish otolith classes**.  
The dataset includes representatives such as:  

*Alepocephalus bicolor, Apistus carinatus, Brachypterois serrulata, Chlorophthalmus acutifrons, Choridactylus multibarbus, Coryphaenoides sp., Cubiceps baxteri, Dactyloptena orientalis, Dactyloptena papilio, Dactyloptena tiltoni, Ectreposbastes imus, Grammoplites suppositus, Hoplostethus sp., Lepidotrigla spiloptera, Minous dempsterae, Minous inermis, Minous trachycephalus, Neomerinthe erostris, Parascombrops pellucidus, Platycephalus indicus, Polymixia fusca, Psenopsis sp., Pterygotrigla arabica, Pterygotrigla hemisticta, Pterygotrigla macrorhynchus, Satyrichthys laticeps, Setarches guentheri, Sorsogona tuberculata, Synagrops japonicus,* and *Uranoscopus sp.*  

### Dataset Samples  

Below are representative otolith images from the dataset: 

| <img src="./cnn/assets/DTR05.png" alt="Dactyloptena tiltoni" width="400"/> | <img src="./cnn/assets/AJ 1 1.6.png" alt="Pterygotrigla macrorhynchus" width="400"/> |
|:------------------------------------------:|:------------------------------------------:|
| **Dactyloptena tiltoni**                   | **Pterygotrigla macrorhynchus**                         |

| <img src="./cnn/assets/Api 2 dorsal1.6x.png" alt="Apistus carinatus" width="400"/> | <img src="./cnn/assets/Chlorophthalmus acutifrons_L_sp3(6).png" alt="Chlorophthalmus acutifrons" width="400"/> |
|:------------------------------------------:|:------------------------------------------:|
| **Apistus carinatus**                       | **Chlorophthalmus acutifrons**                    |
---

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

## Stage II — OOD Robustness and Generalization  

To assess the **robustness of the baseline CNN models (Stage I)** under **open-set conditions**, we performed **out-of-distribution (OOD) detection**. The implementation uses **softmax confidence scoring** provided in [`cnn/core/ood.py`](./cnn/core/ood.py).  

---

### OOD Detection Protocol  

The OOD evaluation requires three inputs:  

1. **ID test set directory** — contains in-distribution otolith images.  
2. **OOD root directory** — contains multiple OOD subsets (e.g., deepsea, shallow marine, freshwater).  
3. **Checkpoint directory** — stores the finetuned CNN model weights.  

Example command:  

```bash
python cnn/scripts/ood_eval.py \
  --id-dir <path_to_test_set> \
  --ood-root <path_to_ood_subsets> \
  --ckpt-dir <path_to_model_checkpoint>
```
### OOD Detection Results  

All metrics and reports are saved in the corresponding timestamped output folder:  
[`cnn/outputs/<timestamp>/models/`](./cnn/outputs/models/)  

The outputs include:  
- 📊 **Confidence histograms** (ID vs OOD distributions)  
- 📈 **ROC and PR curves** for OOD detection performance  
- 🧮 **Evaluation metrics**: AUROC, AUPR (In/Out), FPR  
- 📂 **Per-subset statistics** for each OOD category (e.g., *deepsea*, *shallow marine*, *freshwater*)  
 




