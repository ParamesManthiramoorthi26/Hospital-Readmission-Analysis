# Part 4 – LLM-Powered Model Prediction Explanation Pipeline (Track C)


**Project:** Medical Insurance Readmission Prediction using Machine Learning and Large Language Models  
**Part:** Part 4 – LLM-Powered Feature  
**Track Selected:** Track C – Model Prediction Explanation Pipeline

---

# Project Overview

This project extends the machine learning model developed in Part 3 by integrating a Large Language Model (LLM) to generate human-readable explanations for model predictions.

The trained model (`best_model.pkl`) is loaded using Joblib. Three sample patient records are selected from the encoded dataset. For each sample:

- The machine learning model predicts the class.
- Prediction probability is calculated.
- Feature values, predicted class, and probability are sent to an LLM through the OpenRouter API.
- The LLM returns a structured JSON explanation.
- The JSON response is validated using a predefined JSON Schema.

A PII (Personally Identifiable Information) guardrail is also implemented before every LLM request.

---

# Objectives

- Load the trained machine learning pipeline.
- Predict patient readmission status.
- Explain predictions using an LLM.
- Produce structured JSON responses.
- Validate responses using JSON Schema.
- Prevent sensitive information from being sent to the LLM.
- Compare deterministic and creative outputs using different temperature values.

---

# Technologies Used

- Python
- Scikit-learn
- Joblib
- Pandas
- Requests
- OpenRouter API
- JSON
- JSON Schema
- Regular Expressions (Regex)

---

# Workflow

```
Load best_model.pkl
        │
        ▼
Select Patient Records
        │
        ▼
Predict Class
        │
        ▼
Predict Probability
        │
        ▼
Create Prompt
        │
        ▼
PII Guardrail
        │
        ▼
LLM API (OpenRouter)
        │
        ▼
JSON Response
        │
        ▼
JSON Parsing
        │
        ▼
Schema Validation
        │
        ▼
Prediction Explanation
```

---

# LLM API Connection

The OpenRouter API is accessed using the Python `requests` library.

The API key is stored securely in an environment variable.

```python
os.environ["LLM_API_KEY"] = "YOUR_API_KEY"
```

A reusable function named `call_llm()` performs:

- Payload construction
- HTTP POST request
- Error handling
- JSON response extraction

The function was verified using the prompt:

```
Reply with only the word: hello
```

which successfully returned:

```
hello
```

---

# Prompt Engineering

## System Prompt

```
You are an AI assistant that explains machine learning predictions.

You must respond ONLY in valid JSON.

Return exactly these fields:

prediction_label
confidence_level
top_reason
second_reason
next_step
```

The model is instructed not to generate markdown or additional text.

---

## User Prompt Template

```
Patient Feature Values:

{feature_values}

Machine Learning Prediction

Prediction Label:
{prediction}

Prediction Probability:
{probability}

Based on these values, explain the prediction.

Return ONLY valid JSON.
```

---

# Why Temperature = 0?

Temperature controls randomness in language generation.

For this project, **temperature = 0** was selected because the goal is to obtain deterministic, reproducible, and schema-compliant JSON outputs.

A low temperature consistently selects the highest-probability token, making the responses suitable for structured machine learning applications.

---

# Temperature Comparison

| Input | Temperature = 0 | Temperature = 0.7 | Observation |
|-------|-----------------|-------------------|-------------|
| Sample 1 | Stable JSON | Slight wording variation | Output remains correct but wording changes |
| Sample 2 | Stable JSON | Slight wording variation | Higher creativity at 0.7 |
| Sample 3 | Stable JSON | Slight wording variation | Temperature 0 is more deterministic |

## Observation

Temperature = 0 consistently produced structured JSON suitable for automated validation.

Temperature = 0.7 generated similar explanations but with slightly different wording and phrasing.

---

# JSON Schema

The expected response schema contains five required string fields.

| Field | Type |
|---------|------|
| prediction_label | String |
| confidence_level | String |
| top_reason | String |
| second_reason | String |
| next_step | String |

---

# Structured Output Validation

Each LLM response follows the pipeline below:

1. Remove whitespace.
2. Parse using `json.loads()`.
3. Validate using `jsonschema.validate()`.
4. Catch JSON parsing errors.
5. Catch schema validation errors.
6. Return a fallback dictionary if validation fails.

This ensures every explanation conforms to the expected JSON structure.

---

# PII Guardrail

Before sending any prompt to the LLM, a Regular Expression (Regex) check is performed.

Blocked information includes:

- Email addresses
- Phone numbers

If PII is detected, the LLM request is cancelled and the message:

```
Input blocked: PII detected.
```

is displayed.

---

# Guardrail Demonstration

| Test Input | Result |
|------------|--------|
| Prompt containing email address | Blocked |
| Prompt without PII | Allowed |

---

# Three Sample Demonstration

The pipeline was executed on three patient records.

| Sample | Predicted Class | Probability | Validation |
|---------|-----------------|-------------|------------|
| Sample 1 | No Readmission | 0.975 | Pass |
| Sample 2 | No Readmission | 0.950 | Pass |
| Sample 3 | No Readmission | 0.965 | Pass |

All three responses successfully passed JSON Schema validation.

---

# Example LLM Output

```json
{
  "prediction_label": "No Readmission",
  "confidence_level": "High",
  "top_reason": "Low probability of complications.",
  "second_reason": "Patient features indicate a stable health condition.",
  "next_step": "Continue routine follow-up care."
}
```

---

# Files Included

```
Part4/
│
├── Part4.ipynb
├── best_model.pkl
└── README.md

```

---

# Key Features Implemented

- Machine Learning model loading using Joblib
- Prediction and prediction probability generation
- Prompt engineering
- OpenRouter API integration
- Structured JSON output
- JSON parsing
- JSON Schema validation
- PII Guardrail
- Temperature comparison
- End-to-end prediction explanation pipeline

---

# Conclusion

This project successfully demonstrates how a Large Language Model can be integrated with a trained machine learning model to produce human-readable prediction explanations.

The prediction pipeline generates machine learning predictions, constructs structured prompts, invokes the LLM through the OpenRouter API, validates the returned JSON, and ensures that no personally identifiable information is transmitted. The implementation satisfies all Track C requirements and demonstrates a complete, reliable, and reproducible LLM-assisted prediction explanation workflow.