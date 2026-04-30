# Room Occupancy Detection and Counting from Radar Data

This repository contains the final solution for a machine learning competition project developed for the Machine Learning (MLF) course. The primary objective was to design a robust computer vision model capable of estimating the exact number of people (0, 1, 2, or 3) present in a room based solely on 2D radar scans.

Our final solution achieved an accuracy of **~95.74%**, securing a Top 5 placement on the Kaggle Leaderboard.

## Data Pipeline and Preprocessing
The dataset consists of raw 2D radar heatmaps representing spatial reflections. To ensure the models learned true physical features rather than noise, we implemented a strict preprocessing pipeline:

* **Grayscale Conversion:** Images were processed in single-channel format to reduce dimensionality and focus purely on signal intensity.
* **Physical Zero Preservation:** Instead of statistical Z-Score normalization (which destroys the physical baseline of an empty room), pixel values were Min-Max scaled by dividing by `255.0`. This maps the data to a `[0, 1]` range while keeping absolute darkness (empty space) strictly at 0.
* **Label Synchronization:** We developed a custom loading script to resolve a known offset anomaly in the dataset, correctly pairing CSV labels with their respective images using a programmatic `+1` index shift.
* **Validation Split:** The data was partitioned using a strict 90/10 Stratified split, ensuring that the uneven distribution of target classes (e.g., fewer instances of 3 people) remained consistent across both training and validation sets.

## Architecture: The Diverse Ensemble
To maximize generalization, we discarded the single-model approach in favor of a Diverse Ensemble strategy. We trained three distinct Convolutional Neural Network (CNN) architectures, each designed to "look" at the radar data differently:

1. **The Golden Standard (Base Expert):** 
   Utilizes standard 3x3 convolutional filters and a 256-neuron Dense layer. It acts as the balanced baseline, highly effective at recognizing standard human reflection signatures.
2. **The Macro-Expert (Spatial Expert):** 
   Replaces the initial convolutional layers with larger 5x5 filters. This increases the receptive field early in the network, allowing it to detect broader spatial anomalies and scattered multi-person clusters that smaller filters might miss.
3. **The Deep Analyst (Deep Expert):** 
   Uses 3x3 filters but features a massively expanded fully connected block (512 neurons) combined with aggressive Dropout (0.6). This model acts as the logical tie-breaker for highly ambiguous scans.

## Training Methodology
Preventing overfitting on the radar noise was our primary challenge. We utilized the following techniques during the training phase:

* **Balanced Class Weights:** The model loss function was penalized dynamically. Misclassifying underrepresented classes (like 2 or 3 persons) resulted in a higher error penalty, forcing the network to learn all classes equally.
* **Dynamic Learning Rate:** Implemented `ReduceLROnPlateau` to monitor the validation loss. If the model stopped improving for 5 epochs, the learning rate was halved, allowing the optimizer to settle into narrower local minima.
* **Time-Machine Early Stopping:** We set a generous patience of 18 epochs to allow models to escape temporary plateaus. Once training terminated, the `restore_best_weights=True` callback automatically reverted the network to the exact epoch where it achieved its lowest validation loss.

## Inference Strategy: Test-Time Augmentation (TTA)
Our submission pipeline employs Test-Time Augmentation combined with Soft Voting to achieve maximum robustness on unseen Kaggle test data:

1. **Mirrored Views:** For every single test image, we generate a second, horizontally flipped version. This is physically safe as radar room reflections are generally horizontally symmetrical.
2. **Soft Voting:** The 3 expert models predict probabilities for both the original and the flipped image. 
3. **Probability Averaging:** Instead of hard majority voting, the 6 resulting probability vectors (3 models x 2 views) are summed together. The final class prediction is determined by the `argmax` of these combined probabilities. This allows a highly confident model to overrule slightly confused models.

## Results and Evaluation
The ensemble demonstrates exceptional stability across all classes. Below is the Validation Confusion Matrix, which visualizes the ensemble's performance on the 10% hold-out data.

![Confusion Matrix](pictures/confusion_matrix.png)

**Key Observations from Validation:**
* **Empty Room (0 Persons):** Near-perfect classification. The physical zero baseline successfully prevents false positives from background noise.
* **1 Person:** Extremely high recall. Single reflection clusters are easily isolated by the CNNs.
* **2 vs. 3 Persons:** As expected with radar scattering, the highest rate of ambiguity occurs when multiple people overlap in the radar's line of sight, occasionally causing 3 people to look like a larger cluster of 2. The spatial 5x5 filter expert specifically helps mitigate this issue.

## Authors
* **Team JDVL**
