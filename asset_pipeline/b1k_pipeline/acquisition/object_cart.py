import numpy as np
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By

import csv
import json
import matplotlib.pyplot as plt
import time

START = 500
END = 1000

links = []
with open(r"objlinks.csv", "r") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        link = row["purchase_link"]
        if link.startswith("https"):
            links.append(link.strip())
        elif link.lower() != "true" and link.lower() != "false" and link.lower() != "":
            print(f"Skipping line {i}: {link} due to unknown format.")

links = links[START:END]
print(f"Read {len(links)} lines.")

driver = webdriver.Chrome(
    executable_path=r"C:\Users\capri28\Downloads\chromedriver_win32\chromedriver"
)
driver.get("https://www.turbosquid.com/MemberInfo/")
driver.implicitly_wait(10)

print("Waiting for user to log in and go to Account Info.")
WebDriverWait(driver, 300).until(
    EC.title_contains("Member Info")
)  # Check for login to happen and user to go to

bought_links = {x: 0 for x in links[:START]}
for i, link in enumerate(links):
    # Load the page
    driver.get(link)
    WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CLASS_NAME, "addToCartBtn"))
    )

    # Try to get the price.
    try:
        price_box = driver.find_element_by_id("product-price")
        price = float(price_box.text)
    except:
        try:
            free_box = driver.find_element_by_id("free-price")
            price = 0
        except:
            print(f"Something went wrong with item {i}: {link}. No price tag.")
            continue

    # Store the price.
    bought_links[link] = price

    # Skipping already-bought link.
    if link not in bought_links:
        continue

    time.sleep(3)

    # If it's not free, let's buy it.
    if price > 0:
        button = driver.find_element_by_class_name("addToCartBtn")
        button.click()
        WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.ID, "quickPurchaseModalLabel"))
        )
        print(f"Added {link}, {i + 1}/{len(links)}")
    else:
        print(f"Skipping free {link}, {i + 1}/{len(links)}")

    time.sleep(3)

with open(r"objprices.json", "w") as f:
    json.dump(bought_links, f)

x = np.array(list(bought_links.values()))
print(f"Saved prices. Total: {np.sum(x)}")

q25, q75 = np.percentile(x, [25, 75])
bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
bins = round((x.max() - x.min()) / bin_width)
plt.hist(x, bins=bins)
plt.show()

driver.close()
