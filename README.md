# Dynamic Author-Persona Performed Exceedingly Rapidly (DAPPER) Topic Model #

![DAPPER](../master/docs/images/dapper.png?raw=true)

## Introduction ##
See /docs/dap_2018_arxiv.pdf for technical information on the dynamic author-persona topic model (DAP).

## Getting Started ##
1. Clone the repo:

   ```bash
   cd ~
   git clone https://github.com/robert-giaquinto/dapper.git
   ```

2. Virtual environments.

    It may be easiest to install dependencies into a virtual environment. For python 3+ run:

   ```bash
   cd dapper
   python -m venv ./venv
   ```

   To activate the virtualenv run:

   ```bash
   source ~/dapper/venv/bin/activate
   ```

3. Installing the necessary python packages.

   A requirements.txt file, listing all packages used for this project is included in the repository. To install them first make sure your virtual environment is activated, then run the following line of code:

   ```bash
   pip install --upgrade pip
   ```
   ```bash
   pip install -r ~/dapper/requirements.txt
   ```

4. Install dapper package.

    This is done to allow for absolute imports, which make it easy to load python files can be spread out in different folders. To do this navigate to the `~/dapper` directory and run:

   ```bash
   python setup.py develop
   ```

5. Preparing data for the model

   TODO: build tutorial for easily accessible dataset

6. Running the model

   See /scripts/ for examples of running the model and setting various model parameters.


7. Getting data

   See http://research.signalmedia.co/newsir16/signal-dataset.html for the signalmedia dataset. Codes for preprocessing this data are included in src/experiments/preprocess_signalmedia.py

## Project Structure ##
* `docs/` - Documentation on the model, including derivation and papers related to the model.
* `log/` - Log files from running the programs.
* `scripts/` - Bash scripts for running programs.
* `src/` - Directory containing various sub-packages of the project and any files shared across sub-packages.
