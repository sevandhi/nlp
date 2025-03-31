from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

app = Flask(__name__)

# Load Pre-trained NER Model (XLM-RoBERTa)
model_name = "xlm-roberta-large-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Allowed entity types with CSS classes
ENTITY_COLORS = {
    "PER": "PER",  # Person (Red)
    "LOC": "LOC",  # Location (Green)
    "ORG": "ORG",  # Organization (Blue)
    "MISC": "MISC" # Miscellaneous (Yellow)
}

# Function to extract and highlight entities in text
def highlight_entities(text):
    ner_results = ner_pipeline(text)
    highlighted_text = ""
    last_idx = 0

    for entity in ner_results:
        entity_label = entity["entity"].split("-")[-1]  # Extract entity type
        if entity_label not in ENTITY_COLORS:
            continue  # Ignore unwanted entities

        start, end = entity["start"], entity["end"]
        word = entity["word"].replace("▁", " ")  # Fix subword tokenization spacing

        # Add normal text before the entity
        highlighted_text += text[last_idx:start]

        # Highlight entity
        highlighted_text += f'<span class="{ENTITY_COLORS[entity_label]}">{word}</span>'
        last_idx = end  # Move pointer

    # Add remaining text after last entity
    highlighted_text += text[last_idx:]

    return highlighted_text

@app.route("/", methods=["GET", "POST"])
def index():
    highlighted_paragraph = ""
    paragraph = ""

    if request.method == "POST":
        paragraph = request.form["paragraph"]
        highlighted_paragraph = highlight_entities(paragraph)

    return render_template("index.html", highlighted_paragraph=highlighted_paragraph, paragraph=paragraph)

if __name__ == "__main__":
    app.run(debug=True)