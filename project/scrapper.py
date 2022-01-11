import re
import requests
import json
import time
import glob
import unidecode
import html

subscription_key = "fca1ffbacd424663a47f230d6890c7c3"
search_url = "https://api.bing.microsoft.com/v7.0/news/search"
headers = {"Ocp-Apim-Subscription-Key": subscription_key}

search_term = ""
sites_bait = ["buzzfeednews.com", "upworthy.com", "viralnova.com",
              "BoredPanda.com", "Thatscoop.com", "thepoliticalinsider.com", "Examiner.net"]

sites_n_bait = ["nytimes.com", "washingtonpost.com",
                "theguardian.com", "Bloomberg.com", "Reuters.com"]


def search(keyword: str, site: str, size: int = 1000):
    offset = 0
    params = {"q": "{} site:{}".format(
        keyword, site), "textDecorations": False, "textFormat": "HTML", "count": min(size - offset, 100), "offset": offset}
    titles = []
    while offset < size:
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        search_results = response.json()
        titles += [article for article in search_results["value"]]
        offset += 100
        time.sleep(1)
    return titles


# for i in sites_bait:
#     ts = search("", i, 1000)
#     with open("bait/{}.json".format(i), "w+") as f:
#         print(ts)
#         s = json.dumps(ts)
#         f.write(s)


# for i in sites_n_bait:
#     ts = search("", i, 1000)
#     with open("not_bait/{}.json".format(i), "w+") as f:
#         print(ts)
#         s = json.dumps(ts)
#         f.write(s)

bait = glob.glob("bait/*.json")
n_bait = glob.glob("not_bait/*.json")

bait_list = []
for f in bait:
    with open(f) as fo:
        j = json.loads(fo.read())
        l = [[unidecode.unidecode(html.unescape(x["name"])), 1] for x in j]
        bait_list += l
        del j


n_bait_list = []
for f in n_bait:
    with open(f) as fo:
        j = json.loads(fo.read())
        l = [[unidecode.unidecode(html.unescape(x["name"])), 0] for x in j]
        n_bait_list += l
        del j

with open("bait.json", "w") as f:
    l = json.dumps(bait_list)
    f.write(l)

with open("not_bait.json", "w") as f:
    l = json.dumps(n_bait_list)
    f.write(l)
