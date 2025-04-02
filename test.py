import torch
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load the fine-tuned model and tokenizer
model_path = "fine_tuned_distilbert"  # Update if saved elsewhere
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()  # Set model to evaluation mode

# Load CSV file
csv_file = "export.csv"  # Update filename if needed
df = pd.read_csv(csv_file)

# Ensure required column exists
if "Requirement Text" not in df.columns:
    raise ValueError("CSV file must contain a 'Requirement Text' column.")

# Get the first 10 rows
df_test = df

# Function to predict FR or NFR
def predict_requirement(text):
    encoding = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    
    return "FR" if prediction == 0 else "NFR"

# Apply prediction
df_test["Predicted Type"] = df_test["Requirement Text"].apply(predict_requirement)

# Display results
print(df_test[["Requirement Text", "Predicted Type"]])

# Save the predictions to a new CSV file
df_test.to_csv("predicted_requirements.csv", index=False)
print("Predictions saved to predicted_requirements.csv")
