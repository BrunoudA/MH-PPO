# Repository for MH-PPO and Coop-MH-PPO: A hybrid action space and multi-agent model for autonomous vehicle (AV) control during interactions with pedestrians.

## 1) Structure:
  - python files: MH-PPO and Coop-MH-PPO models (scalable model is learning with variable numbers of agents)
    
  - Environments folder for MDP environment with GYM library
    
  - load_model folder for trained model


## 2) Create Virtual Environment:
  ```python -m venv myenv ```
  
  For Windows
  ```myenv\Scripts\activate ```
  
  For macOS/Linux
  ```source myenv/bin/activate ```
  
  Install dependencies: Install the required libraries using:
  ```pip install -r requirements.txt ```

## 3) Run algorithms:
  Recommendation: Use Jupyter Notebook
  
  Else transform the jupyter file with the command:
  
  ```jupyter nbconvert --to script file.ipynb```
  
  then 
  
  ```python file.py```
  
