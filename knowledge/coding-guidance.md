# Coding Guidance for INFO 4940/5940

## Programming Language Choice

Students can complete all assignments in either R or Python. The course provides parallel instruction in both languages.

## Python Setup & Best Practices

### Environment Setup
- Use Python 3.9 or higher
- Work in Posit Workbench IDE or local environment
- Use virtual environments for package management
- Install packages via pip or conda

### Essential Python Libraries for ML

**Data Manipulation:**
```python
import pandas as pd  # DataFrames and data manipulation
import numpy as np   # Numerical computing
```

**Machine Learning:**
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
```

**Visualization:**
```python
import matplotlib.pyplot as plt
import seaborn as sns
```

**LLM Integration:**
```python
from openai import OpenAI  # OpenAI API
import anthropic           # Claude API
```

### Python Data Workflows

**Loading Data:**
```python
# CSV files
df = pd.read_csv('data/file.csv')

# Feather files (efficient binary format)
df = pd.read_feather('data/file.feather')

# Excel files
df = pd.read_excel('data/file.xlsx')
```

**Basic Data Exploration:**
```python
df.head()              # First 5 rows
df.info()              # Column types and null counts
df.describe()          # Summary statistics
df.shape               # Dimensions (rows, columns)
df.columns             # Column names
df.isnull().sum()      # Count missing values
```

**Data Cleaning:**
```python
# Drop missing values
df_clean = df.dropna()

# Fill missing values
df['column'] = df['column'].fillna(0)

# Filter rows
df_filtered = df[df['column'] > 10]

# Select columns
df_subset = df[['col1', 'col2', 'col3']]
```

### Machine Learning Workflows

**Train/Test Split:**
```python
from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Model Training & Evaluation:**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print(classification_report(y_test, y_pred))
```

### Working with LLMs in Python

**OpenAI API:**
```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain machine learning"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

**Structured Outputs:**
```python
from pydantic import BaseModel

class Classification(BaseModel):
    category: int
    confidence: float

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    response_format=Classification
)

result = response.choices[0].message.parsed
```

**Batch Processing:**
```python
# Create batch file with requests
batch_requests = []
for item in data:
    request = {
        "custom_id": str(item['id']),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": item['text']}],
            "max_tokens": 100
        }
    }
    batch_requests.append(request)

# Submit batch
with open("batch_input.jsonl", "w") as f:
    for req in batch_requests:
        f.write(json.dumps(req) + "\n")

batch = client.batches.create(
    input_file_id=file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)
```

### Visualization Best Practices

**Matplotlib:**
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Line 1')
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Plot Title')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('figures/plot.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Seaborn:**
```python
import seaborn as sns

# Heatmap
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')

# Distribution plot
sns.histplot(data=df, x='column', bins=30)

# Correlation matrix
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
```

## R Setup & Best Practices

### Essential R Libraries

**Tidyverse:**
```r
library(tidyverse)  # Includes dplyr, ggplot2, tidyr, readr, etc.
```

**Machine Learning:**
```r
library(tidymodels)  # ML workflows
library(caret)       # Classification and regression training
```

**Visualization:**
```r
library(ggplot2)
library(gt)  # Professional tables
```

### R Data Workflows

**Loading Data:**
```r
# CSV files
df <- read_csv("data/file.csv")

# Feather files
library(arrow)
df <- read_feather("data/file.feather")
```

**Data Manipulation:**
```r
df %>%
  filter(column > 10) %>%
  select(col1, col2, col3) %>%
  mutate(new_col = col1 * 2) %>%
  group_by(category) %>%
  summarize(mean_value = mean(col1))
```

## Common Debugging Tips

**Python:**
- Check data types: `df.dtypes`
- Inspect shape: `df.shape`
- Look for NaN: `df.isnull().sum()`
- Print intermediate results
- Use try/except for error handling

**R:**
- Check structure: `str(df)`
- View data: `View(df)`
- Check class: `class(object)`
- Use `print()` statements
- Leverage `tryCatch()` for error handling

## Reproducibility

**Set Random Seeds:**
```python
# Python
import random
import numpy as np
random.seed(42)
np.random.seed(42)
```

```r
# R
set.seed(42)
```

## Version Control with Git

**Basic Workflow:**
```bash
git status                    # Check current status
git add file.py              # Stage changes
git commit -m "Add feature"  # Commit with message
git push                     # Push to remote
```

**Best Practices:**
- Commit frequently with meaningful messages
- Don't commit API keys or sensitive data
- Use .gitignore for large files and credentials
- Make at least 3 commits per assignment

## Getting Help

1. **Read error messages carefully** - They often tell you exactly what's wrong
2. **Check documentation** - Official docs are your best friend
3. **Search online** - Stack Overflow, GitHub issues
4. **Ask in discussion forums** - Your classmates may have similar questions
5. **Attend office hours** - Course staff can provide personalized help
6. **Use Ezra** - The course AI assistant
