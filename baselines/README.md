Here are the steps to run different models for clinical trial outcome prediction:

1. **Running SPOT**
   - Update the train, test, and validation data paths in `run_spot.py`.
   - Execute the Python file:
     ```bash
     python run_spot.py
     ```

2. **Running BioBERT**
   - Modify the train, test, and validation data paths in `biobert_trial_outcome.py`.
   - Run the Python file:
     ```bash
     python biobert_trial_outcome.py
     ```

3. **Running PubMedBERT**
   - Change the train, test, and validation data paths in `pubmedbert_trial_outcome.py`.
   - Execute the Python file:
     ```bash
     python pubmedbert_trial_outcome.py
     ```

4. **Running SVM, XGBoost, MLP, RF, or LR**
   - Ensure that the paths in `baselines.py` are correct.
   - Run the Python file:
     ```bash
     python baselines.py
     ```
