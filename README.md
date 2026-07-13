# Attention-based-RL-MIL
This repository implements an Attention-based Reinforcement Learning Multiple Instance Learning (RL-MIL) framework. The core framework, including MIL and RL-MIL with Epsilon-Greedy baselines, is directly adopted from the repository associated with a previous master's thesis. This README details the specific modifications made to integrate two novel hypernetwork frameworks.

# Project Scope and Baselines
This project extends the existing RL-MIL framework by focusing on the integration and evaluation of debiasing mechanisms. The MIL model and the RL-MIL with Epsilon-Greedy models serve as baselines, consistent with the original repository. However, neither attention-based models nor non-aggregated bags are needed for this analysis. Consequently, the dataset distinction and all attention-based RL policies have been excluded.

All RL-MIL models have been developed and tested specifically with the following configurations: 
- task_type="classification"
- rl_task_model="vanilla"
- sample_algorithm="without_replacement"
- prefix="loss"
- rl_model="policy_only"
- reg_alg="sum"

Other prefixes are retained for experimental purposes. 

# Significant Codebase Adjustments
The following list details the significant changes made to the original repository:  
- moved scripts into src/ folder to use module-based python
- split models.py into several files in src/models/ for quick access to specific model parts
- implemented debiasing models using gradient reversal and hypernetwork logic
- extracted parts of run_rlmil.py into trainer classes that include all training logic specific to each of the 3 frameworks
- created run_trainer.py for running all frameworks from one shell script (run_trainer.sh)
- configs.py: added specific configuration flags for hypernetwork architectures, including: embedding_dim, fourier_scale and hyper_ratio.

# Step-by-Step Usage Guide
Follow these steps to set up and run the attention-based RL-MIL framework:
1. Create a Python virtual environment ('venv') using python 3.12 or newer.
2. Install all necessary dependencies using the requirements.txt file. 
3. Optional: Connect your Weights & Biases (wandb) account for experiment tracking. (https://wandb.ai/site)
2. Download OULAD data: Inside the data/ directory, create a new folder named 'raw'. Download all OULAD files from https://analyse.kmi.open.ac.uk/open-dataset and place them into the data/raw/ folder.
3. Initial data Exploration and Cleaning: Run the eda_OULAD.ipynb Jupyter Notebook. This notebook performs exploratory data analysis, checks and imputes null values, handles duplicates, and prepares the data for merging. All cleaned and prepared files will be saved in a new data/clean/ folder. 
4. Create Data Bags: Execute createbags_aggregated.py. This creates "raw" and "encoded" bags by summing VLE clicks per activity type, providing aggregated instances. The raw bags are primarily for inspection and understanding the instance structure. The encoded bags are used for model training.
5. Prepare Data for Model Input: Run prepare_oulad_data.py. This script loads the previously created bags, pads them with zero's up to a fixed maximum number of instances per bag, adds a mask to distinguish real instances from padding, and generates a trian/validation/test split. If you want to run multiple seeds, you can also use the create_datasets.sh shell script.
6. Run MIL Model (Baseline): The MIL model is typically run before the RL-MIL models to find the best configurations per pooling technique. Adjust and execute scripts/run_mil.sh. You can tune parameters like random seed and pooling techniques. Runs are saved in the /run folder, and logs can be found in the /logs folder.
7. Run the RL-MIL Models: After running the MIL baseline, you can execute the run_trainer.sh script for the RL-MIL models.
8. Analyze results: The src/results/ folder contains scripts for consolidating experimental outcomes, creating tables, visualizations, and calculating interpretability metrics. Start with gather_results.py to collect data from you model runs. The comments at the top of the files include their specific purpose.

**Note**: If you want to run these experiments on the Snellius national supercomputer, you need to edit the shell scripts to include the required partition and load the neccessary modules for python 3.12 or newer during virtual environment creation and after loading the virtual environment.
