from pdf_reader import load_folder
from prompt_chain import kategorizer_chain, kategorizer_big_chain
import json
from dotenv import load_dotenv
import tiktoken  # Add this import

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


def count_tokens(text, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


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


def kategorize_folder_in_one(path):
    pubmed, pdfs = load_folder(path)
    data = []
    matching_data = []

    """# die pubmed liste soll in 10 Teile aufgeteilt werden
    pubmed = [pubmed[i : i + 10] for i in range(0, len(pubmed), 10)]
    print(pubmed)"""

    pubmed_string = "\n---\n".join(pubmed)
    token_count = count_tokens(pubmed_string)
    print(f"Token count for pubmed_string: {token_count}")
    pubmed_data = kategorizer_big_chain.invoke({"context": pubmed_string})
    print(pubmed_data)
    # for every entries in the pubmed_data dict, add it to the data list
    for key, value in pubmed_data.items():
        data.append({key: value})
        if check_values({key: value}):
            matching_data.append({key: value})
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
