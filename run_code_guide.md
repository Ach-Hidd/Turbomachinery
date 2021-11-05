# Run code locally

Install Anaconda and create environment (This only has to be done once):

1. Download and install Anaconda (Python3.8) on your PC: [Anaconda Downloads Page](https://www.anaconda.com/products/individual#Downloads)

2. Download  {download}`environment.yml <environment.yml>`.
3. Once installed open ``Anaconda Prompt``: A terminal window will appear with the base environment activated.
4. Navigate to the folder where the ``environment.yml`` file is located using the {command}`cd` command:
    ```
    cd \NAVIGATE_TO_FOLDER_ON_PC_WHERE_ENVIRONMENT.YML_IS_LOCATED
    ```

5. Create the environment from the ``environment.yml`` file:
    ```
    conda env create -f environment.yml
    ```
6. Once the installation of the environment is finished. Activate the environment (turbo_a is the name of the environment):
    ```
    conda activate turbo_a
    ```
7. Run the following command to open Jupyter lab:
    ```
    jupyter lab
    ```

To start an assignment (assignment 1 is used in the following example):

8. Download {download}`assignment1.ipynb <assignment1.ipynb>` (or use the download button at the top of the assingment page).

9. Navigate to the location on your PC where ``assignment1.ipynb`` is located and open the notebook.

10. Good work, you are good to go! 
