from pdf_reader import load_folder
from prompt_chain import kategorizer_chain
import json
from dotenv import load_dotenv

load_dotenv()


def check_values(json_obj):
    # Prüfe für jeden Key im JSON Object
    required_values = {
        "peer_review": ["yes"],
        "about_parents": ["yes"],
        "published_since_2014": ["yes"],
        "migration_immigration_asylum_refugee": ["yes"],
        "transnational_parenthood_or_families": ["yes"],
        "research_type": ["qualitative", "quantitative"],  # Multiple valid values
    }

    for key, value in json_obj.items():
        # Prüfe ob alle required_values im value-dict übereinstimmen
        if isinstance(value, dict):
            matches = all(
                value.get(req_key) in req_values
                for req_key, req_values in required_values.items()
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
        print("Check\n")
        data.append(pubmed_data)
        if check_values(pubmed_data):
            matching_data.append(pubmed_data)

    # Sortiere die matching_data Liste alphabetisch nach dem ersten Key der Objekte
    matching_data.sort(key=lambda x: list(x.keys())[0])
    # Sortiere die data Liste alphabetisch nach dem ersten Key der Objekte
    data.sort(key=lambda x: list(x.keys())[0])

    # Speichere die gefilterten Ergebnisse
    if matching_data:
        with open("matching_documents.json", "w", encoding="utf-8") as f:
            json.dump(matching_data, f, ensure_ascii=False, indent=2)
    with open("documents.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data


if __name__ == "__main__":
    data = kategorize_folder("Datein")
    print(data)
