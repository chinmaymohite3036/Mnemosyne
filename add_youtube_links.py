import json
import os

VIDEO_URLS = {
    "01": "https://youtu.be/tVzUXW6siu0",
    "02": "https://youtu.be/kJEsTjH5mVg",
    "03": "https://youtu.be/BGeDBfCIqas",
    "04": "https://youtu.be/nXba2-mgn1k",
    "05": "https://youtu.be/1BsVhumGlNc",
    "06": "https://youtu.be/CyRlWlaJnTY",
    "07": "https://youtu.be/tLBlhp0SA_0",
    "08": "https://youtu.be/vnnlUCLfn6I",
    "09": "https://youtu.be/vlAWzsGd-Yk",
    "10": "https://youtu.be/XZwBNDGuWGU",
    "11": "https://youtu.be/fhoDRB53DwY",
    "12": "https://youtu.be/5xFRg_TzlAg",
    "13": "https://youtu.be/cvsbHZcDx8w",
    "14": "https://youtu.be/1dkfuga2_Ps",
    "15": "https://youtu.be/-XwZpYIyCEA",
    "16": "https://youtu.be/anGMeDGvZhw",
    "17": "https://youtu.be/1cEG1T8beO4",
    "18": "https://youtu.be/Xrxd6cEajhM"
}

for filename in os.listdir("newjsons"):

    if not filename.endswith(".json"):
        continue

    filepath = os.path.join("newjsons", filename)

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    for chunk in data["chunks"]:

        number = chunk["number"].zfill(2)
        start = int(chunk["start"])

        youtube_url = VIDEO_URLS[number]

        chunk["youtube_url"] = f"{youtube_url}?t={start}"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Updated: {filename}")

print("\nAll JSON files updated successfully!")