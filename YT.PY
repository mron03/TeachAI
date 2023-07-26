from youtube_transcript_api import YouTubeTranscriptApi

# retrieve the available transcripts
transcript_list = YouTubeTranscriptApi.list_transcripts('cMEA2usAZBc')

# iterate over all available transcripts
for transcript in transcript_list:

    transcript = transcript.fetch()
    res = ''

    for t in transcript:
        res = res + t['text'] + ' '

print(res)
# # you can also directly filter for the language you are looking for, using the transcript list
# transcript = transcript_list.find_transcript(['de', 'en'])  

# # or just filter for manually created transcripts  
# transcript = transcript_list.find_manually_created_transcript(['de', 'en'])  

# # or automatically generated ones  
# transcript = transcript_list.find_generated_transcript(['de', 'en'])