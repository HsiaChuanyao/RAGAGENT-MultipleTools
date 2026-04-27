import requests
from bs4 import SoupStrainer
from langchain_community.document_loaders import WebBaseLoader
from requests.adapters import HTTPAdapter
from urllib3 import Retry

session = requests.Session()
retry_strategy = Retry(
    total=5,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
)

adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

urls = ["https://docs.langchain.com/oss/python/learn", "https://docs.langchain.com/oss/python/langchain/rag","https://docs.langchain.com/oss/python/langgraph/overview"]
bs4_strainer = SoupStrainer(class_=("post-title","post-header","post-content"))
web_loader = WebBaseLoader(
    web_path = urls,
    bs_kwargs={"parse_only":bs4_strainer},
    requests_kwargs={
        "timeout": 10,
        "headers": {
            "User_Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; X64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36)"
        }
    }
)

file_content = web_loader.load()

print(file_content)