# 1) Structure:
  - python files: MH-PPO and Coop-MH-PPO models (scalable model is learning with variable numbers of agents) 
  - Environments folder for MDP environment with GYM library
  - load_model folder for trained model


# 2) Create Virtual Environment:
  ```bash python -m venv myenv ```
  For Windows
  ```bash myenv\Scripts\activate ```
  For macOS/Linux
  ```bash source myenv/bin/activate ```
  
  Install dependencies: Install the required libraries using:
  ```bash pip install -r requirements.txt ```

# 3) Run algorithm:
  ```bash python file.py (e.g. Coop-MH-PPO-scalable.py)```
  
