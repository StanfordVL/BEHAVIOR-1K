import csv
import matplotlib.pyplot as plt
import urllib.parse
from collections import defaultdict

links = []
data = defaultdict(list)
with open(r"objlinks.csv", "r") as f:
    reader = csv.reader(f)

    # This skips the first row of the CSV file.
    next(reader)

    for i, row in enumerate(reader):
        obj = row[0]
        link = row[7]
        if link.startswith("https"):
            parsed = urllib.parse.urlparse(link)
            sanitized = "https://www.turbosquid.com" + parsed.path
            data[sanitized].append(obj.strip().replace(" ", "_"))
        elif link.lower() != "true" and link.lower() != "false" and link.lower() != "":
            print(f"Skipping line {i}: {link} due to unknown format.")

print(f"Read {len(data)} URLs.")

with open("link_to_obj.csv", "w", newline="") as csvfile:
    fieldnames = ["link", "objs"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    writer.writeheader()

    for link, objs in data.items():
        writer.writerow({"link": link, "objs": ",".join(objs)})

print("Saved mapping.")
