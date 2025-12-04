# Prediction Agent

This agent predicts machine failure using a trained ML model.

### Input
- Temperature
- Pressure
- Vibration
- Sound level

### Output
- Failure risk: 0 (No risk) or 1 (Likely failure)

### How it works
1. Takes user input from UI.
2. Passes it to model.pkl file using predict() function.
3. Returns the prediction to Flask frontend.

