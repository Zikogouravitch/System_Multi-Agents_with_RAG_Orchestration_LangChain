import requests
from bs4 import BeautifulSoup
import time
import json
import sys
import os

folder = "data_raw"

if not os.path.exists(folder):
    os.makedirs(folder)

print("Folder ready")

def scrape_letter(letter):
    output_file = f"data_raw/data_raw_{letter}.jsonl"

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    page = 1
    count = 0
    log_file = "summary.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        pass
    
    while True:
        if page == 1:
            url = f"https://medicament.ma/listing-des-medicaments/?lettre={letter}"
        else:
            url = f"https://medicament.ma/listing-des-medicaments/page/{page}//?lettre={letter}"

        print(f"[{letter}] Page {page}")

        try:
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code != 200:
                break

            soup = BeautifulSoup(res.text, "html.parser")
            items = soup.find_all("li", class_="listing-item")

            if not items:
                break

            for item in items:
                try:
                    name = item.find("p", class_="primary").text.strip()
                    link = item.find("a")["href"]

                    print(f" -> {name}")

                    detail_res = requests.get(link, headers=headers, timeout=10)

                    if detail_res.status_code == 200:
                        detail_soup = BeautifulSoup(detail_res.text, "html.parser")
                        details_block = detail_soup.find("div", class_="medicine-details")

                        raw_data = {
                            "drug_name": name,
                            "details": {}
                        }

                        if details_block:
                            lines = details_block.find_all("div", class_="detail-item")

                            for line in lines:
                                key = line.find("div", class_="detail-header").text.strip()
                                value = line.find("div", class_="detail-content").text.strip()

                                raw_data["details"][key] = value

                        with open(output_file, "a", encoding="utf-8") as f:
                            json.dump(raw_data, f, ensure_ascii=False)
                            f.write("\n")
                        count += 1
                    time.sleep(2)

                except Exception as e:
                    print(f"[!] erreur médicament: {e}")

            page += 1
            time.sleep(2)

        except Exception as e:
            print(f"[!] erreur connexion: {e}")
            break

    print(f"Done: {output_file}")
    print(f"\n Lettre '{letter}' terminée")
    print(f" Nombre total de médicaments récupérés: {count}")
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"Lettre: {letter} | Count: {count} | File: {output_file}\n")


if __name__ == "__main__":
    letter = sys.argv[1]
    scrape_letter(letter)