#!/usr/bin/env python3

import os
import time
import argparse
from multiprocessing import Pool
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
import yt_dlp as youtube_dl

def scrape_soundcloud_links(tag, num_scrolls, path_to_chrome_driver):
    """
    Scrape SoundCloud links based on a given tag.
    
    :param tag: The tag to search for on SoundCloud
    :param num_scrolls: Number of times to scroll the page
    :param path_to_chrome_driver: Path to the Chrome driver executable
    :return: A set of unique SoundCloud track URLs
    """
    options = Options()
    options.add_argument("--headless")  # Run in headless environment
    options.add_argument("--no-sandbox")  # Do not sandbox Chrome
    options.add_argument("--disable-dev-shm-usage")  # Prevent issues with shared memory

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
        time.sleep(5)  # Wait for content to load

        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == previous_height:
            retries += 1
            if retries > 5:  # Stop if no new content after 5 tries
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
    """
    Download a SoundCloud track.
    
    :param track_url: URL of the track to download
    :param download_folder: Folder to save the downloaded track
    """
    # Extract the artist name and song name from the URL
    url_parts = track_url.strip().split('/')
    artist_name = url_parts[-2]
    song_name = url_parts[-1]

    # Construct the output template
    output_template = os.path.join(download_folder, f'{song_name} - {artist_name}.%(ext)s')

    # Check if the file already exists
    existing_files = [f for f in os.listdir(download_folder) if f.startswith(f'{song_name} - {artist_name}')]
    if existing_files:
        print(f"Skipping download for {track_url}: File already exists")
        return

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
    parser = argparse.ArgumentParser(description="Scrape and download SoundCloud tracks based on a tag.")
    parser.add_argument('input_file', type=str,
                        help="The file to save the scraped links to (e.g., 'breakcore_links.txt')")
    parser.add_argument('chrome_driver_path', type=str,
                        help="Relative path to the chrome driver")
    parser.add_argument('output_dir', type=str, help="Path to where the mp3 files will be saved")
    parser.add_argument('--processes', type=int, default=6, help="Number of download processes")
    tag = 'breakcore'
    num_scrolls = 15
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Scrape links if the input file doesn't exist
    if not os.path.exists(args.input_file):
        links = scrape_soundcloud_links(tag, num_scrolls, args.chrome_driver_path)
        with open(args.input_file, 'w') as file:
            for link in links:
                file.write(link + '\n')
        print(f'Scraped {len(links)} links for tag "{tag}" and saved them to {args.input_file}')
    else:
        print(f'Using existing links from {args.input_file}')

    # Read links from the input file
    with open(args.input_file, 'r') as file:
        links = [link.strip() for link in file.readlines() if link.strip()]

    # Download tracks using multiple processes
    with Pool(processes=args.processes) as pool:
        pool.starmap(download_soundcloud_track, [(link, args.output_dir) for link in links])

if __name__ == "__main__":
    main()
