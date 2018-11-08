# Guide on how to set up work environment

## Environment - Conda

 - Conda YML : using the existing environment.yml can create conda env through

    ```bash
    conda env create -f environment.yml
    ```


## Setting up Core Packages

To install local packges, run the following commands (from current directory):

 - For base halite code

    ```bash
    cd halite && python setup.py install
    ```

 - For haliterl extension

    ```bash
    cd haliterl && python setup.py install
    ```

## Anything Else:

 - Multiagent (open ai gym)

    ```bash
    git clone https://github.com/openai/multiagent-particle-envs.git
    cd multiagent-particle-envs && pip install -e .
    ```

## Linting

 - vscode : shift - command - p, python : select intepreter and find environment