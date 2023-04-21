<p align="center">
  <img src="https://d2k0ddhflgrk1i.cloudfront.net/Websections/Huisstijl/Bouwstenen/Logo/02-Visual-Bouwstenen-Logo-Varianten-v1.png"/><br>
  <br><br>
</p>

# Group D2: Machine Learning in Bioinformatics (CS4260) Project
The codebase represents group efforts towards reproducing several figures from the inDelphi research paper, as well as individual research related to the paper. While writing the code, we tried to match it to the authors' work while adapting it for the dataset given. Importantly, we left comments in the code explaining our understanding of the authors' approach and how the code relates to the supplementary materials (example: **training_model.py**). Explanations on how to reproduce the results from the report can be found further in readme. Exact reproduction of our efforts is possible due to the utilization of fixed randomisation seeds throughout the codebase.

## Authors

Francesco Piccoli (ID: 5848474)

Boriss Bērmans (ID: 4918673)

Sirada Kaewchino (ID: 117922132)

Wessel Oosterbroek (ID: 4961544) 

## User Manual

How to setup repo:
```
**macOS/Linux**
git clone https://github.com/francescoopiccoli/MLBio_project.git
cd MLBio_project
conda create --name ML_bio_project_env python=3.10  
conda activate ML_bio_project_env       
pip install -r requirements.txt
```

In case `pip install -r requirements.txt` fails due to the package 'sklearn' being deprecated, do the following in the terminal:

```
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
```

Then, run the pip command again.

## Structure
The codebase is structured as follows:
- `figures/` contains the reproduced figures, as well as an extra figure visualizing differences between observed and predicted frequencies for more selected outcomes;
- `input/` has example count and deletion feature files, as well as the main pickle file used for training the model **(counts + del_features)**;
- `model-mlbio/` has weights from the 2 neural networks, as well resulting files from running the KNN model. They are there for convenience (to be able to generate the figures right away), but can be regenerated by running the `training_model.py` script;
- `mylib/` contains helper functions from the authors of inDelphi (necessary to train the model);
- `output_test/` contains target sequences from the test split (necessary to generate one of the figures with `generate_figures.py`);
- `research-questions/` folder contains the scripts to reproduce the individual RQ results;
- `grna-libA.txt` contains the gRNA sequences for each target sequence in libA;
- `inDelphi.py` contains the logic for calculating predictions given our trained inDelphi model;
- `knn_helper.py` has the code for training the KNN model;
- `generate_figures.py` is a script for generating the `1e` and `3f` figures;
- `nn_helper.py` contains helper functions for the neural networks;
- `targets-libA.txt` contains the libA target sequences;
- `training_model.py` is a script for training the complete inDelphi model;
- `utilities.py` contains shared utility functions.

## How to generate the figures selected for the reproduction? 
To generate the figures, run the following command from the root directory:

```
python -W ignore generate_figures.py
```

The `-w ignore` flag is necessary due to pandas warnings that we could not easily solve. These warnings do not affect the results.

## How to train the model?
To train the model, run the following command from the root directory:

```
python -W ignore training_model.py
```

This command is going to generate weights for the neural networks and resulting files from the KNN model under the `outputaab/` directory. The `-w ignore` flag is necessary due to the warnings from deprecated packages. To reproduce the results accurately, we had to use the packages of the same versions as in authors' code.

## How to generate predictions for a single target sequence?
Create the following python script:
```
import inDelphi

# Pass a sample target sequence and the respective cutsite
pred_df, stats = inDelphi.predict('GTCAAAGTCCATTATTCGAAATCCAATCCAAGGTCACTGGAATTTGTTGATTAAG', 27)
```

## How to reproduce the individual results?

### Boriss Bermans
To reproduce the results, simply run the following command within the `research-questions` directory:

```
python boris-rq.py
```

This script should generate a folder structure `output-boris-rq/ten-fold-crossvalidation` inside `research-questions`. It will contain the logs of each fold, for each model, as well as the saved weights after 50 epochs for each fold. When the crossvalidation ends, it generates statistics and displays the plots that can be found in the report. Here is the total output of running the script:

- Histogram and PDF of fold differences
- Mean, STD, median and skew of fold differences
- Sign test results
- Wilcoxon test results