# DAPPER Topic Model #

<img src="../master/docs/images/dapper.png" width="100">

## Dynamic Author-Persona Performed Exceedingly Rapidly ##

The DAPPER topic model is designed for multi-author corpora in which authors write over time. Unlike other temporal topic models, DAPPER doesn't model the change in language associated with a topic (e.g. Dynamic Topic Models) but instead models the trajectory of topics that an author discusses over time.


## Introduction ##
See [/docs/dap_2018_arxiv.pdf](https://github.com/robert-giaquinto/dapper/blob/master/docs/dap_2018_arxiv_acm.pdf) for technical information on the dynamic author-persona topic model (DAP).

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

   See [Signal Media 1M](http://research.signalmedia.co/newsir16/signal-dataset.html) to download the Signal Media dataset.

   See `/src/preprocessing/preprocess_signalmedia.py` for tools to prepare the Signal Media data. Or use the already preprocessed data included in this repository.


6. Running the model

   See /scripts/ for examples of running the model and setting various model parameters.



## Project Structure ##
* `docs/` - Documentation on the model, including derivation and papers related to the model.
* `log/` - Log files from running the programs.
* `scripts/` - Bash scripts for running programs.
* `src/` - Directory containing various sub-packages of the project and any files shared across sub-packages.
