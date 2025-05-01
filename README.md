# Manifold Learning on Proteins
This project is uses Variational Autoencoders for Manifold Learning on proteins with the express goal being to generate latent spaces which simpler models can use to perform tasks such as protein family classification. The Proteinshake Dataset will be downloaded when any of the scipts is first executed (and won't need to be redownloaded once done).

There are three different models:
- Basic VAE
- Point VAE
- Graph VAE

These are defined in the models folder in src.
The each model has its own defined hyperparameter sweep file outlined in src/training.
These can be run from terminal with the following command "python -m src.training.*"
where * is the model sweep file name.
When run, the program will ask for a dataset input, the dataset used for this project is "Pfam", it will also ask for a name for the sweep folder in which the training logs and trained models will be stored.


For the Beta experiments the same procedure can be used but from the src/experiment_training folder using 
"python -m src.experiment_training.*" Similar to above, it will ask for a dataset, write "Pfam".
Then it will ask for experiment type. There are two options, the first is to run the Beta experiment with a set model and varying regularisation coefficients and the other is to train a single "final model". 
To run the beta experiment input "Beta", train the final model input "Final_Model" (this is important as it will save the train and validation datasets used).

The src/dataset_classes contains the pytorch dataset classes that handle data pre-processing and retrieving samples for batches.

utils contains other functions mostly related to data handling.

Notebooks contains different jupyter notebooks used for the project to create graphs and visualisations. Notably, 
- the CLUSTER_ANALYSIS notebook contains the final evaluation of generated latent spaces for the models.
- Beta_Exp_visualisation contains the generation of plots for the beta experiment.
- Sequence comparison contains code for applying the Smith-Watermna Algorithm to generated sequences

For reproducibility:
    - Run Sweep and use the visualisation notebook found in experiments/training logs to visualise different behaviours (or use tensorboard)
    - Run the beta experiment script with the parameters of the best model from the sweep
        - Use Beta_Exp_visualisation to visualise latent space using 2D PCA
    - Based on the above adjust Beta to reflect and run the Final Model function from the relevant experiment training script with the relevant parameters.
    - For the use of VAEs for visualisation run the same script as above but with the latent dimensions set to two.
Then the notebooks mentioned above can be used for visualisation.