from pdf_reader import load_folder
from prompt_chain import kategorizer_chain
import json


def check_values(json_obj):
    # Prüfe für jeden Key im JSON Object
    required_values = {"peer_review": "yes", "about_parents": "yes"}

    for key, value in json_obj.items():
        # Prüfe ob alle required_values im value-dict übereinstimmen
        if isinstance(value, dict):
            matches = all(
                value.get(req_key) == req_value
                for req_key, req_value in required_values.items()
            )
            if matches:
                return True
    return False


def kategorize_folder(path):
    pubmed, pdfs = load_folder(path)
    data = []
    matching_data = []

    for pdf in pdfs:
        pdf_data = kategorizer_chain.invoke({"context": pdf})
        data.append(pdf_data)
        if check_values(pdf_data):
            matching_data.append(pdf_data)

    for record in pubmed:
        pubmed_data = kategorizer_chain.invoke({"context": record})
        data.append(pubmed_data)
        if check_values(pubmed_data):
            matching_data.append(pubmed_data)

    # Speichere die gefilterten Ergebnisse
    if matching_data:
        with open("matching_documents.json", "w", encoding="utf-8") as f:
            json.dump(matching_data, f, ensure_ascii=False, indent=2)

    return data


if __name__ == "__main__":
    data = kategorize_folder("Datein")
    print(data)
