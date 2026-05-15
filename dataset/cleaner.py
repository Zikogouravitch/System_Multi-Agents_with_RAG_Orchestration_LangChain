import json
import glob
import re

def safe_float(value):
    try:
        return float(value.split()[0].replace(",", "."))
    except:
        return None

def normalize(input_pattern, output_file):
    files = glob.glob(input_pattern)

    print(f"Files found: {len(files)}")

    with open(output_file, "w", encoding="utf-8") as out:

        for file in files:
            print(f"Processing: {file}")

            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    d = item.get("details", {})

                    data = {
                        "drug_name": item.get("drug_name"),
                        "presentation": d.get("Présentation"),
                        "manufacturer": d.get("Distributeur ou fabriquant"),
                        "composition": [],
                        "dosage": {},
                        "therapeutic_class": d.get("Classe thérapeutique"),
                        "status": d.get("Statut"),
                        "atc_code": d.get("Code ATC"),
                        "price": {},
                        "table": d.get("Tableau"),
                        "indications": [],
                        "product_type": d.get("Nature du Produit"),
                        "extra": {}
                    }

                    if d.get("Composition"):
                        data["composition"] = [x.strip() for x in d["Composition"].split("|")]

                    if d.get("Dosage"):
                        parts = d["Dosage"].split("|")
                        for i, p in enumerate(parts):
                            data["dosage"][f"substance_{i+1}"] = p.strip()

                    if d.get("PPV"):
                        data["price"]["ppv"] = safe_float(d["PPV"])

                    if d.get("Prix hospitalier"):
                        data["price"]["hospital"] = safe_float(d["Prix hospitalier"])

                    if d.get("PPC"):
                        data["price"]["ppc"] = safe_float(d["PPC"])

                    data["price"]["currency"] = "MAD"

                    if d.get("Indication(s)"):
                        data["indications"] = [
                            x.strip("- ").strip()
                            for x in re.split(r"\r|\n", d["Indication(s)"])
                            if x.strip()
                        ]

                    for key, value in d.items():
                        k = key.lower()
                        v = str(value).lower()

                        if "psychoactive" in k:
                            data["psychoactive"] = "oui" in v

                        if "dépendance" in k or "dependance" in k:
                            data["dependence_risk"] = "oui" in v

                    # extra fields
                    known_fields = [
                        "Présentation", "Dosage", "Distributeur ou fabriquant",
                        "Composition", "Classe thérapeutique", "Statut",
                        "Code ATC", "PPV", "Prix hospitalier", "PPC",
                        "Tableau", "Indication(s)", "Nature du Produit"
                    ]

                    for key, value in d.items():
                        if key not in known_fields:
                            data["extra"][key] = value

                    indications_text = ", ".join(data["indications"]) if data["indications"] else "usage non spécifié"
                    composition_text = ", ".join(data["composition"]) if data["composition"] else ""

                    data["text"] = f"{data['drug_name']} est un médicament. Composition: {composition_text}. Indications: {indications_text}."

                    if not data["drug_name"]:
                        continue

                    json.dump(data, out, ensure_ascii=False)
                    out.write("\n")

    print(f"\nAll clean data saved in: {output_file}")


if __name__ == "__main__":
    normalize("data_raw/data_raw_*.jsonl", "medicaments.jsonl")