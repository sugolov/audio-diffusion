from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import argparse
import time


def scrape_soundcloud_links(tag, num_scrolls):
    options = Options()
    options.headless = False  # Change to True if you want to run in headless mode
    service = Service('/home/alex/Downloads/chromedriver-linux64/chromedriver-linux64/chromedriver')
    driver = webdriver.Chrome(service=service, options=options)

    search_url = f'https://soundcloud.com/tags/{tag}/popular-tracks'
    driver.get(search_url)

    links = set()
    wait = WebDriverWait(driver, 10)

    previous_height = driver.execute_script("return document.body.scrollHeight")
    retries = 0

    for _ in range(num_scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)  # Increase this if content loads slowly

        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == previous_height:
            retries += 1
            if retries > 5:  # If the page height hasn't changed after 5 tries, stop
                print("No more content to load, stopping...")
                break
        else:
            retries = 0  # Reset retries if new content is loaded

        previous_height = new_height

        anchors = driver.find_elements(By.CSS_SELECTOR, 'a[href*="/"]')
        for anchor in anchors:
            href = anchor.get_attribute('href')
            if 'soundcloud.com' in href and 'tags' not in href:
                links.add(href)

    driver.quit()
    return links


def main():
    parser = argparse.ArgumentParser(description="Scrape SoundCloud popular tracks based on a tag.")
    parser.add_argument('tag', type=str, help="The tag to use for searching SoundCloud (e.g., 'breakcore')")
    parser.add_argument('output_file', type=str,
                        help="The file to save the scraped links to (e.g., 'breakcore_links.txt')")
    parser.add_argument('--num_scrolls', type=int, default=10,
                        help="The number of times to scroll down the page (default is 10)")

    args = parser.parse_args()

    links = scrape_soundcloud_links(args.tag, num_scrolls=args.num_scrolls)

    # Save the links to the specified file
    with open(args.output_file, 'w') as file:
        for link in links:
            file.write(link + '\n')

    print(f'Scraped {len(links)} links for tag "{args.tag}" and saved them to {args.output_file}')


if __name__ == "__main__":
    main()
