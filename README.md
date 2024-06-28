# Project Description
10.5281/zenodo.12192518

Development of an Artificial Neural Network to Identify Immune Cell–Cancer Gene Set Interactions

Sreeja Appala

6/27/2024

# Code

All the code was conducted in Python with TensorFlow sequential class.

# Artificial Neural Network (ANN) Development

First, a general ANN was developed, including all necessary parameters ('1setup.py'). Then, each parameter was tested individually for its optimal value ('2architecture.py' to '9activation.py'). This was done by iterating the model through different values for that parameter while keeping the rest of the model constant. The optimal value was chosen based on the best evaluation metrics from the model within those runs: the lowest MSE, the highest R-squared value, and the highest accuracy.

'finalmodel.py' contains the code for the final artificial neural network (ANN) incorporating all the optimal parameters found earlier. 'output.csv' includes all the predictors and responses that the model trains and validates on, with features being immune cell fractions and targets being gene set enrichment levels. The model achieved an MSE of 0.0035, an R-squared value of 0.99, and an accuracy of 96%. It successfully predicted the relationship between the immune cell fractions and gene set enrichment levels. Later, sensitivity analysis was applied to reveal which immune cells were impacting each of the gene sets.

# Features and Targets

Feature 1 - Memory B cells

Feature 2 - Plasma cells

Feature 3 - CD4+ T cells

Feature 4 - M2 macrophages

Feature 5 - Mast cells

Feature 6 - Neutrophils

Target 1 - Angiogenesis

Target 2 - Hedgehog signaling

Target 3 - Epithelial–mesenchymal transition (EMT)

Target 4 - Apical junction

Target 5 - TGF-beta signaling

# Research Paper

Soon to come, will be linked here.
