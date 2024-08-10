#!/usr/bin/env python3

import os
import csv
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import yt_dlp

# standardize mp3 filenames 
def sanitize_filename(title):
    # remove invalid characters and limit length
    title = re.sub(r'[\\/*?:"<>|]', "", title)
    return title[:200] # limit to 200 chars to avoid potential issues with very long filenames


# get yt video id and title for comparisons and filenames, respectively
def get_video_info(url):
    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
        info = ydl.extract_info(url, download=False)
        return info['id'], info['title']

# given a yt video's url, download it as an mp3 file
def download_audio(url, output_dir, existing_ids):
    try:
        video_id, video_title = get_video_info(url)
        # do not download duplicates (if id already exists)
        if video_id in existing_ids: 
            return f"Skipped (already exists): {url}"


        safe_title = sanitize_filename(video_title)
        filename = f"{video_id} - {safe_title}"
        output_path = os.path.join(output_dir, filename)

        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': output_path,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return f"Successfully downloaded: {url}"
    except Exception as e:
        return f"Error downloading {url}: {str(e)}"

# read the contents of a csv or textfile into a python list of strings
def read_links(file_path):
    links = []
    if file_path.endswith('.csv'):
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            links = [row[0] for row in reader if row]
    else:
        with open(file_path, 'r') as txtfile:
            links = [line.strip() for line in txtfile if line.strip()]
    return links

def get_existing_ids(output_dir):
    return {filename.split(' - ')[0] for filename in os.listdir(output_dir) if filename.endswith('.mp3')}

# driver program 
def main(file_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    links = read_links(file_path)
    existing_ids = get_existing_ids(output_dir)
    
    # execute the scraping concurrently
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(download_audio, url, output_dir, existing_ids): url for url in links}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result()
                print(result)
            except Exception as exc:
                print(f'{url} generated an exception: {exc}')

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download YouTube videos as MP3 from a file of links.")
    parser.add_argument("file_path", help="Path to the text or CSV file containing YouTube links")
    parser.add_argument("output_dir", help="Directory to save the downloaded MP3 files")
    
    args = parser.parse_args()
    
    main(args.file_path, args.output_dir)
