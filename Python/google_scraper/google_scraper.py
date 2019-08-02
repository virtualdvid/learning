# based on the code from Rushil Srivastava
# https://github.com/rushilsrivastava/image_search/

import os
import sys
import requests
import time
import urllib
import argparse
import json
import secrets
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from pathlib import Path
from lxml.html import fromstring
from fake_useragent import UserAgent
from webdriver_manager.chrome import ChromeDriverManager #https://github.com/SergeyPirogov/webdriver_manager

def search(keyword):
    base_url = "https://www.google.com/search?q={}&source=lnms&tbm=isch&sa=X&ved=0ahUKEwiwoLXK1qLVAhWqwFQKHYMwBs8Q_AUICigB"

    url = base_url.format(keyword.lower().replace(" ", "+"))

    # Create a browser and resize for exact pinpoints
    browser = webdriver.Chrome(ChromeDriverManager().install())
    browser.set_window_size(1024, 768)
    print("\n===============================================\n")
    print("[%] Successfully launched Chrome Browser")

    # Open the link
    browser.get(url)
    time.sleep(1)
    print("[%] Successfully opened link.")

    element = browser.find_element_by_tag_name("body")

    print("[%] Scrolling down.")
    # Scroll down
    for i in range(30):
        element.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.3)  # bot id protection

    try:
        browser.find_element_by_id("smb").click()
        print("[%] Successfully clicked 'Show More Button'.")
        for i in range(50):
            element.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.3)  # bot id protection
    except Exception:
        for i in range(10):
            element.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.3)  # bot id protection

    print("[%] Reached end of Page.")

    time.sleep(1)

    # Get page source and close the browser
    source = browser.page_source
    browser.close()
    print("[%] Closed Browser.")

    return source


def download_image(link, image_data, query):
    download_image.delta += 1
    # Use a random user agent header for bot id
    ua = UserAgent()
    headers = {"User-Agent": ua.random}

    # Get the image link
    try:
        # Get the file name and type
        file_name = link.split("/")[-1]
        type = file_name.split(".")[-1]
        type = (type[:3]) if len(type) > 3 else type
        if type.lower() == "jpe":
            type = "jpeg"
        if type.lower() not in ["jpeg", "jfif", "exif", "tiff", "gif", "bmp", "png", "webp", "jpg"]:
            type = "jpg"

        image_name = "{}.{}".format(str(secrets.token_hex(9)), type)
        status = {'label':query,
            'image':str(link),
            'image_name':image_name,
            'status':None}
        # Download the image
        print("[%] Downloading Image #{} from {}".format(download_image.delta, link))
        try:
            urllib.request.urlretrieve(link,
                                       "images/{}/{}".format(query, image_name))
            print("[%] Downloaded File")
            status['status'] = 'success'
        except Exception as e:
            download_image.delta -= 1
            print("[!] Issue Downloading: {}\n[!] Error: {}".format(link, e))
            status['status'] = 'failed'
    except Exception as e:
        download_image.delta -= 1
        print("[!] Issue getting: {}\n[!] Error:: {}".format(link, e))
        status['status'] = 'failed'

    return status


def scrape(keyword_list, limit=500, json_name='data'):
    json_name = 'json/{}.json'.format(json_name)
    for keyword in keyword_list:
        print("\n===============================================\n")
        print("[%] {} started.".format(keyword))
        print("\n===============================================\n")

        # set stack limit
        sys.setrecursionlimit(1000000)

        # get user input and search on google
        query = keyword.lower().replace(" ", "_")

        if not os.path.isdir("images/{}".format(query)):
            os.makedirs("images/{}".format(query))

        source = search(keyword)

        # Parse the page source and download pics
        soup = BeautifulSoup(str(source), "html.parser")
        ua = UserAgent()
        headers = {"User-Agent": ua.random}

        # Get the links and image data
        links = soup.find_all("a", class_="rg_l")

        if not os.path.isfile(json_name):
            with open(json_name, 'w') as file:
                json.dump([], file)

        # Clip Limit
        # if len(links) > limit:
        #     links = links[0:limit]
        
        print("[%] Indexed {} Images.".format(len(links)))
        print("\n===============================================\n")
        print("[%] Getting Image Information.\n")
        image_data = None
        download_image.delta = 0
        
        for i, a in enumerate(soup.find_all("div", class_="rg_meta")):
            r = requests.get("https://www.google.com" + links[i].get("href"), headers=headers)
            title = str(fromstring(r.content).findtext(".//title"))
            link = title.split(" ")[-1]

            with open(json_name) as file:
                try:
                    used_links = json.load(file)
                    used_links_check = [(link['image'], link['status']) for link in used_links if link['label'] == query]
                    for used_link_img, used_link_status in used_links_check:
                        if used_link_img == link and used_link_status == 'success' or len(used_links) > limit:
                            # print("\n[%] Already downloaded")
                            continue    
                except:
                    print("[%] Not content in json file.\n")
                    used_links = []

            print("\n[%] Getting info on: {}".format(link))
            try:
                image_data = "google", query, json.loads(a.text)["pt"], json.loads(a.text)["s"], json.loads(a.text)["st"], json.loads(a.text)["ou"], json.loads(a.text)["ru"]
                status = download_image(link, image_data, query)
                used_links.append(status)
            except Exception as e:
                print("[!] Issue getting data: {}\n[!] Error: {}".format(image_data, e))
                continue

            with open(json_name, 'w') as outfile:
                json.dump(used_links, outfile)
                print('[%] Loaded to json\n')
            
            if len(used_links_check) >= limit:
                break

        print("[%] Downloaded {} images for label {}.".format(download_image.delta, keyword))
