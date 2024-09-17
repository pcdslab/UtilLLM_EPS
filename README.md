# System Requirements
System with Ubuntu 20.04 or later with at least 4 CPU cores, atleast 1 GPU, 64GBs of memory and greater than 30GB of available storage is recommended. Other Python 3.12.0 or higher and its built-in libraries. All needed packages ae detailed in the `requirments.txt` file and can be downloaded following the setup instructions below.

# Setup

1. **Check the pip Version**:
   - Verify if pip is installed by running the following command:
     ```sh
     pip --version
     ```
   - If pip **is installed**, you should see an output similar to:
     ```console
     ppara014@dragon:~/CDMA_Folder/Code$ pip --version
     pip 24.0 from /usr/lib/python3.12/site-packages/pip (python 3.12)
     ```
   - If pip is installed, you can skip to step 3.

2. **Install pip** (if not installed):
   - If you receive an error or no output when checking the pip version, install pip by running:
     ```sh
     sudo apt-get install python-pip
     ```
   - (You may need sudo permissions if you're working on a shared resource.)

3. **Verify if venv is Installed**:
   - Ensure that the `venv` module is available by running:
     ```sh
     python3 -m venv --help
     ```
   - If `venv` **is available**, you should see an output similar to:
     ```console
     ppara014@dragon:~/CDMA_Folder/Code$ python3 -m venv --help
     usage: venv [-h] [--system-site-packages] [--symlinks | --copies] [--clear] [--upgrade] [--without-pip] [--prompt PROMPT] [--upgrade-deps] ENV_DIR [ENV_DIR ...]

     Creates virtual Python environments in one or more target directories.

     positional arguments:
     ENV_DIR               A directory to create the environment in.

     options:
     -h, --help            show this help message and exit
     --system-site-packages
                           Give the virtual environment access to the system site-packages dir.
     --symlinks            Try to use symlinks rather than copies, when symlinks are not the default for the platform.
     --copies              Try to use copies rather than symlinks, even when symlinks are the default for the platform.
     --clear               Delete the contents of the environment directory if it already exists, before environment creation.
     --upgrade             Upgrade the environment directory to use this version of Python, assuming Python has been upgraded in-place.
     --without-pip         Skips installing or upgrading pip in the virtual environment (pip is bootstrapped by default)
     --prompt PROMPT       Provides an alternative prompt prefix for this environment.
     --upgrade-deps        Upgrade core dependencies (pip) to the latest version in PyPI

     Once an environment has been created, you may wish to activate it, e.g., by sourcing an activate script in its bin directory.
     ```
   - If `venv` is installed, skip to step 5.

4. **Install venv** (if not available):
   - If `venv` is not available or returns an error, install it by running:
     ```sh
     sudo apt install python3.12-venv
     ```
   - (Sudo permissions may be required.)

5. **Create a Virtual Environment**:
   - Create a virtual environment named `myenv` (you can choose a different name if preferred) by running:
     ```sh
     python3 -m venv myenv
     ```

6. **Activate the Virtual Environment**:
   - Activate the virtual environment by running:
     ```sh
     source myenv/bin/activate
     ```

7. **Upgrade pip**:
   - Update pip to the latest version by running:
     ```sh
     python -m pip install --upgrade pip
     ```

8. **Download the Repository**:
   - Clone the repository to your local machine and navigate to the project directory by running:
     ```sh
     git clone https://github.com/pcdslab/UtilLLM_EPS.git
     cd UtilLLM_EPS
     ```
   - For more information, see [managing repositories](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).

9. **Install Required Packages**:
   - Install the necessary Python packages by running:
     ```sh
     pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
     python3 -m pip install -r requirements.txt
     ```

You are now ready to retrain the models and generate results.

# Retraining Models

1. **Data Acquisition**:
   - To create dataset please see [MLSPred-Bench](https://github.com/pcdslab/MLSPred-Bench).
   - To get processed dataset, please send an email request to **[saeed.researchlab@gmail.com]**

2. **Data Preparation**:
   - Once you have received the data, place the training data in the `data/training_data/` directory and the validation data in the `data/validation_data/` directory.

3. **Retraining All Models on All Benchmarks**:
   - To retrain all models across all benchmarks, execute the following command:
     ```bash
     python3 fine_tune_models.py
     ```

4. **Retraining a Single Model on a Specific Benchmark**:
   - To retrain a single model on a particular benchmark, use the command:
     ```bash
     python3 fine_tune_models.py --model_name <name> --BM <num>
     ```
   - Here, `<name>` can be one of the following: `swinv2`, `mit`, `longformer`.
   - `<num>` corresponds to any benchmark number from `01` to `12`.

5. **Post-Training Outputs**:
   - Upon completion of training:
     - The best model checkpoint and the last model checkpoint will be saved in the `finetuned_model/<name>/` directory.
     - A CSV file containing the predicted probabilities, true labels, and for Longformer, additional majority-vote true labels, will be stored in `results/<name>_custom/`.

# Generating Results

1. **Generating ROC Curves and ROC-AUC Score Table**:
   - To recreate the ROC curves and the ROC-AUC score table, run the following command:
     ```bash
     python3 generate_roc_auc_graph_table.py
     ```
   - After the script completes:
     - The `roc_auc_scores.xlsx` file will contain the ROC-AUC score table.
     - The `ROC_AUC_CURVE_BM1To12.eps` file will contain the ROC curves.

2. **Generating Accuracy, Sensitivity, and Specificity Table**:
   - To recreate the accuracy, sensitivity, and specificity table, run:
     ```bash
     python3 generate_acc_sen_spe_table.py
     ```
   - After the script completes, the `model_performance.xlsx` file will contain the accuracy, sensitivity, and specificity table.