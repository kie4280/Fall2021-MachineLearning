import re
import requests
import json

subscription_key = "fca1ffbacd424663a47f230d6890c7c3"
search_url = "https://api.bing.microsoft.com/v7.0/news/search"
headers = {"Ocp-Apim-Subscription-Key": subscription_key}

search_term = ""
sites_bait = ["buzzfeednews.com", "upworthy.com", "viralnova.com",
              " BoredPanda.com", "Thatscoop.com", "thepoliticalinsider.com", "Examiner.net"]

sites_n_bait = ["nytimes.com", "washingtonpost.com",
                "theguardian.com", "Bloomberg.com", "Reuters.com"]


def search(keyword: str, site: str, size: int = 1000):

    params = {"q": "{} site:{}".format(
        keyword, site), "textDecorations": False, "textFormat": "HTML", "count": 100}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    titles = [article["name"] for article in search_results["value"]]
    return titles


i = sites[1]

ts = search("", i)
with open("{}.json".format(i), "w+") as f:
    print(ts)
    s = json.dumps(ts)
    f.write(s)
