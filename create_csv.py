import pandas as pd

# Sample dataset
data = {
    "text": [
        "The patient experiences auditory hallucinations and visual delusions.",
        "There is a noticeable disorganized thinking pattern in the patient's speech.",
        "The patient shows diminished emotional expression and reduced motivation.",
        "Social withdrawal and lack of interest in daily activities are evident in the patient.",
        "The patient has difficulties with attention and memory deficits.",
        "There are significant impairments in the patient's executive function and problem-solving skills.",
        "The patient reports seeing and hearing things that are not there.",
        "The patient exhibits flat affect and speaks in a monotone voice.",
        "The patient struggles with planning and organizing tasks, indicating impaired executive function.",
        "Hallucinations and delusions are persistent in the patient's behavior.",
        "The patient shows a lack of motivation and avoids social interactions.",
        "There are noticeable deficits in the patient's memory and reasoning abilities.",
        "The patient believes in things that are clearly not true and has disorganized speech.",
        "The patient has a reduced ability to express emotions and appears apathetic.",
        "The patient has difficulty concentrating and remembering information."
    ],
    "label": [
        "Positive Symptoms",
        "Positive Symptoms",
        "Negative Symptoms",
        "Negative Symptoms",
        "Cognitive Impairment",
        "Cognitive Impairment",
        "Positive Symptoms",
        "Negative Symptoms",
        "Cognitive Impairment",
        "Positive Symptoms",
        "Negative Symptoms",
        "Cognitive Impairment",
        "Positive Symptoms",
        "Negative Symptoms",
        "Cognitive Impairment"
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('/Users/nityasridhar/Documents/schizophrenia_symptoms.csv', index=False)
