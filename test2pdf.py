import re

def is_youtube_link(link):
    # Regular expression pattern to match YouTube video URLs
    youtube_pattern = r'^https?://(?:www\.)?(?:youtu\.be/|youtube\.com/watch\?v=)([\w-]+)'

    # Check if the link matches the YouTube pattern
    return re.match(youtube_pattern, link) is not None

# Test the function with some example links
links = [
    "https://www.youtube.com/watch?v=abcdefghijk",
    "https://youtu.be/abcdefghijk",
    "https://www.youtube.com/embed/abcdefghijk",
    "https://www.youtube.com/v/abcdefghijk",
    "https://www.youtube.com/user/username/u/1/watch?v=abcdefghijk",
    "https://www.youtube.com/playlist?list=abcdefghijk",
    "https://www.google.com"  # Not a YouTube link
]

for link in links:
    if is_youtube_link(link):
        print(f"{link} is a YouTube video link.")
    else:
        print(f"{link} is not a YouTube video link.")
