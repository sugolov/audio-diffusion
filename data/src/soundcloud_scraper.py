from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import argparse
import time
import os
import yt_dlp as youtube_dl
import argparse
from multiprocessing import Pool

def scrape_soundcloud_links(tag, num_scrolls, path_to_chrome_driver):
    options = Options()
    options.headless = False  # Change to True if you want to run in headless mode
    service = Service(path_to_chrome_driver)
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

def download_soundcloud_track(track_url, download_folder):
    # Extract the artist name and song name from the URL
    url_parts = track_url.strip().split('/')
    artist_name = url_parts[-2]
    song_name = url_parts[-1]

    # Construct the output template to save the file as "song_name - artist_name.mp3"
    output_template = os.path.join(download_folder, f'{song_name} - {artist_name}.%(ext)s')

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_template,
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([track_url])
        print(f"Successfully downloaded: {track_url}")
    except Exception as e:
        print(f"Failed to download {track_url}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Scrape SoundCloud breakcore tracks based on a tag.")
    parser.add_argument('input_file', type=str,
                        help="The file to save the scraped links to (e.g., 'breakcore_links.txt')")
    parser.add_argument('chrome_driver_path', type=str,
                        help="relative path to the chrome driver")
    parser.add_argument('output_dir', type = str , help = "path to where the mp3 files will be saved")
    parser.add_argument('--processes', type = int, default = 6, help = "number of download processes")
    tag = 'breakcore'
    num_scrolls = 15
    args = parser.parse_args()

    links = scrape_soundcloud_links(tag, num_scrolls, args.chrome_driver_path)

    # Save the links to the specified file
    with open(args.input_file, 'w') as file:
        for link in links:
            file.write(link + '\n')

    print(f'Scraped {len(links)} links for tag "{tag}" and saved them to {args.output_file}')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(args.input_file, 'r') as file:
        links = [link.strip() for link in file.readlines() if link.strip()]
    with Pool(processes=args.processes) as pool:
        pool.starmap(download_soundcloud_track, [(link, args.output_dir) for link in links])

if __name__ == "__main__":
    main()
