import torch
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Load the fine-tuned model and tokenizer
model_path = "fine_tuned_distilbert"  # Update path if different
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()  # Set model to evaluation mode

# Load Excel file
excel_file = "data.xlsx"  # Update if filename is different
df = pd.read_excel(excel_file)

# Ensure required columns exist
if "Requirement Text" not in df.columns or "Type" not in df.columns:
    raise ValueError("Excel file must contain 'Requirement Text' and 'Type' columns.")

# Drop missing values
df = df.dropna(subset=["Requirement Text", "Type"])

# Split into 80% train and 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Function to convert text labels to numerical labels
def label_encoder(label):
    return 0 if label == "FR" else 1  # FR → 0, NFR → 1

# Function to predict FR or NFR
def predict_requirement(text):
    encoding = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    
    return prediction  # Returns 0 (FR) or 1 (NFR)

# Apply prediction on test set
test_df["Actual Label"] = test_df["Type"].apply(label_encoder)
test_df["Predicted Label"] = test_df["Requirement Text"].apply(predict_requirement)
test_df["Predicted Type"] = test_df["Predicted Label"].apply(lambda x: "FR" if x == 0 else "NFR")

# Print first 10 rows of predictions
print(test_df[["Requirement Text", "Type", "Predicted Type"]].head(10))

# Calculate performance metrics
y_true = test_df["Actual Label"].tolist()
y_pred = test_df["Predicted Label"].tolist()

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="weighted")
report = classification_report(y_true, y_pred, target_names=["FR", "NFR"])
conf_matrix = confusion_matrix(y_true, y_pred)

# Print evaluation metrics
print("\n📌 Model Evaluation Results:")
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ F1 Score: {f1:.4f}")
print("\n📌 Classification Report:")
print(report)
print("\n📌 Confusion Matrix:")
print(conf_matrix)

# Save results to an Excel file
test_df.to_excel("test_predictions.xlsx", index=False)
print("📂 Predictions saved to test_predictions.xlsx")
