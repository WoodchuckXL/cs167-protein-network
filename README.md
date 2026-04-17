# CS 167 Network Team
Some python tools for making GO term prediction using protein networks

## Usage Instructions
Start by creating the python virtual environment by running in the project:

    % python3 -m venv .venv
    % source .venv/bin/activate
    % pip install -r requirements.txt

To fetch the STRING data, run

    % python3 fetch_data.py

Then to process the STRING data to get DSD output, adjacency matrix, and GO terms, run

    % python3 networkgraph.py {LINK_FILE} {GO_FILE}

Where the link and go entry files are both in the `data` folder. 
This program results in three csv files in the `results` folder.

To run the ML model on these csv files, run

    % python3 NetworksML.py --adj-path {ADJACENCY_MATRIX_FILE} --go-path {GO_MATRIX_FILE}

This creates descriptive statistics plots, accuracy and ROC curve plots, and a summary report in the `results` folder.