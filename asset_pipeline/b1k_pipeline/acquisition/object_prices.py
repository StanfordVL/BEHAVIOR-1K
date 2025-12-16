import csv
import json
import tqdm
import requests
import bs4
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt

links = []
with open(r"objlinks.csv", "r") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        link = row["purchase_link"]
        if link.startswith("https"):
            links.append(link.strip())
        elif link.lower() != "true" and link.lower() != "false" and link.lower() != "":
            print(f"Skipping line {i}: {link} due to unknown format.")

print(f"Read {len(links)} lines.")


def load_price(link, timeout):
    html = requests.get(link).content
    dom = bs4.BeautifulSoup(html, "html.parser")
    price_thing = dom.find(id="product-price")
    price_text = price_thing.attrs["content"]
    price = float(price_text) if price_text.lower() != "free" else 0
    return price


prices = {}
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    # Start the load operations and mark each future with its URL
    future_to_url = {executor.submit(load_price, link, 10): link for link in links}
    for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_url)):
        link = future_to_url[future]
        try:
            price = future.result()
            prices[link] = price
        except Exception as exc:
            print("%r generated an exception: %s" % (link, exc))

paid_objs = len([price for price in prices.values() if price > 0])
print(f"Total of {paid_objs} non-free objects.")

with open(r"C:\Users\capri28\Downloads\objprices.json", "w") as f:
    json.dump(prices, f)

print("Saved prices.")

x = np.array(list(prices.values()))
q25, q75 = np.percentile(x, [25, 75])
bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
bins = round((x.max() - x.min()) / bin_width)
plt.hist(x, bins=bins)
plt.show()
