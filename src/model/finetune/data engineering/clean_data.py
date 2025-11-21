# File should return a clean wlasl data json file containing actual paths to the videos after processing
# It should do this by:
# 1. Loading the original wlasl data json file
# 2. Filter WLASL entries based on some criteria (e.g., only certain classes)
# 3. For each entry, download the youtube video if not already downloaded
# 4. Extract the relevant clip from the youtube video based on the start and end times
# 5. Save the processed data into a new json file with updated paths to the local video clips called wlasl_cleaned.json