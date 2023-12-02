from googlesearch import search
import requests
from bs4 import BeautifulSoup

def get_search_results(query):
    search_results = search(query, num=5, stop=5, pause=2)
    return search_results

def scrape_webpage(url):
    try:
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        return text
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return None

def main():
    query = "What did Elon Musk said to X advertisers?"
    search_results = get_search_results(query)

    scraped_texts = []
    reference_links = []

    for idx, link in enumerate(search_results, 1):
        text = scrape_webpage(link)
        if text:
            scraped_texts.append(text)
            reference_links.append(f"[{idx}] - {link}")

    # Saving scraped text in a variable
    all_scraped_text = '\n\n'.join(scraped_texts)

    # Saving reference links
    all_reference_links = ', '.join(reference_links)

    print("\nScraped Text:\n")
    print(all_scraped_text)

    print("\nReference Links:\n")
    print(all_reference_links)

if __name__ == "__main__":
    main()
