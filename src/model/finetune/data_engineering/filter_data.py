# File should return a clean wlasl data json file containing actual paths to the videos after processing
# It should do this by:
# 1. Loading the original wlasl data json file
# 2. Filter WLASL entries based on some criteria (e.g., only certain classes)
# 3. For each entry, download the youtube video if not already downloaded
# 4. Extract the relevant clip from the youtube video based on the start and end times
# 5. Save the processed data into a new json file with updated paths to the local video clips called wlasl_cleaned.json
#

import json
from pathlib import Path
import os

#320 word Glossary for sign language recognition
GLOSSARY = [# Pronouns (15)
    "I", "YOU", "HE", "SHE", "THEY", "WE", "ME", "YOU-PLURAL", "THEM",
    "THIS", "THAT", "MINE", "YOURS", "THEIRS", "SELF",

    # Basic Verbs (40)
    "BE", "HAVE", "DO", "GO", "COME", "MAKE", "LOOK", "SEE", "WATCH", "GIVE",
    "TAKE", "TELL", "ASK", "USE", "NEED", "WANT", "KNOW", "THINK", "FEEL",
    "LIKE", "LOVE", "HATE", "EAT", "DRINK", "SLEEP", "WORK", "PLAY", "HELP",
    "MOVE", "STOP", "START", "OPEN", "CLOSE", "PUT", "BRING", "CALL", "WAIT",
    "CHANGE", "FIND",

    # Modal Verbs (10)
    "CAN", "CAN'T", "MUST", "SHOULD", "SHOULD-NOT", "WILL", "WILL-NOT",
    "MAYBE", "TRY", "CONTINUE",

    # Time Words (25)
    "NOW", "TODAY", "TOMORROW", "YESTERDAY", "MORNING", "AFTERNOON",
    "NIGHT", "WEEK", "MONTH", "YEAR", "HOUR", "MINUTE", "SECOND", "SOON",
    "LATE", "EARLY", "ALWAYS", "NEVER", "SOMETIMES", "BEFORE", "AFTER",
    "AGAIN", "FIRST", "LAST", "NEXT",

    # People & Roles (25)
    "PERSON", "MAN", "WOMAN", "BOY", "GIRL", "FRIEND", "FAMILY", "MOTHER",
    "FATHER", "BROTHER", "SISTER", "BABY", "CHILD", "STUDENT", "TEACHER",
    "DOCTOR", "NURSE", "POLICE", "DRIVER", "WORKER", "BOSS", "CUSTOMER",
    "PEOPLE", "GROUP", "TEAM",

    # Places (25)
    "HOME", "SCHOOL", "WORKPLACE", "STORE", "HOSPITAL", "BATHROOM", "KITCHEN",
    "ROOM", "CAR", "BUS", "TRAIN", "STREET", "PARK", "CITY", "COUNTRY",
    "BUILDING", "RESTAURANT", "BANK", "POST-OFFICE", "AIRPORT", "HOUSE",
    "MARKET", "CHURCH", "LIBRARY", "FARM",

    # Objects & Things (30)
    "BOOK", "PHONE", "COMPUTER", "TABLE", "CHAIR", "BED", "CUP", "FOOD",
    "WATER", "CLOTHES", "SHOES", "BAG", "PAPER", "PEN", "MONEY", "KEY",
    "DOOR", "WINDOW", "LIGHT", "FIRE", "TOY", "BALL", "TOOLS", "BOX",
    "BOTTLE", "MEDICINE", "MAP", "CARD", "TV", "CAMERA",

    # Common Nouns (25)
    "DAY", "TIME", "THING", "PLACE", "WAY", "NAME", "QUESTION", "ANSWER",
    "PROBLEM", "IDEA", "REASON", "STORY", "NEWS", "JOB", "PLAN", "GOAL",
    "CHOICE", "SIGN", "LANGUAGE", "COLOR", "NUMBER", "EVENT", "WORLD",
    "BODY", "MIND",

    # Actions / Movement (20)
    "RUN", "WALK", "JUMP", "TURN", "FOLLOW", "LEAVE", "ARRIVE", "ENTER",
    "EXIT", "FALL", "PUSH", "PULL", "CARRY", "LIFT", "DROP", "HIT",
    "TOUCH", "SHAKE", "TURN-ON", "TURN-OFF",

    # Feelings (20)
    "HAPPY", "SAD", "ANGRY", "AFRAID", "SURPRISED", "EXCITED", "NERVOUS",
    "SICK", "HURT", "TIRED", "BORED", "CONFUSED", "CALM", "PROUD", "BRAVE",
    "LONELY", "RELAX", "WORRY", "CARE", "HOPE",

    # Descriptors (35)
    "BIG", "SMALL", "LONG", "SHORT", "FAST", "SLOW", "HOT", "COLD", "NEW",
    "OLD", "GOOD", "BAD", "RIGHT", "WRONG", "EASY", "HARD", "LOUD", "QUIET",
    "FULL", "EMPTY", "CLEAN", "DIRTY", "STRONG", "WEAK", "RICH", "POOR",
    "SAFE", "DANGEROUS", "SAME", "DIFFERENT", "TRUE", "FALSE", "HIGH",
    "LOW", "IMPORTANT",

    # Colors (10)
    "RED", "BLUE", "GREEN", "YELLOW", "BLACK", "WHITE", "ORANGE", "PINK",
    "PURPLE", "BROWN",

    # Quantities (20)
    "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE",
    "TEN", "MANY", "FEW", "MORE", "LESS", "ALL", "NONE", "SOME", "ENOUGH",
    "TOO-MUCH", "TOO-LITTLE",

    # Question Words (10)
    "WHO", "WHAT", "WHERE", "WHEN", "WHY", "HOW", "WHICH", "HOW-MUCH",
    "HOW-MANY", "WHAT-FOR",

    # Connectors (10)
    "AND", "OR", "BUT", "IF", "BECAUSE", "WITH", "WITHOUT", "ABOUT",
    "FOR", "FROM"]

with open("../raw_data/wlasl.json", "r") as file:
  wlasl_data = json.loads(file.read())
# clean data by glossary

class DataCleaner():
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.dataset = []
        if self.verbose:
            print("\n\n🧹 Starting WLASL BASE GLOSSARY FILTERING Process...")
        self.glossary = GLOSSARY

        if self.verbose:
            print(f"\nGlossary size: {len(self.glossary)}")
        self.raw_data = wlasl_data
        self.clean_data = []
        
        # filter data by glossary items 
        self._filter_data_by_glossary()
        self.glossary_clean_data = self.get_clean_data()

        if self.verbose:
            print("\n🧹 Completed WLASL BASE GLOSSARY FILTERING.... ")
            print(f"\nCleaned Glossary Size: {len(self.glossary_clean_data)}")
            print("\n\n📂 Starting Downloaded Video Filteration Process...")

        
       
        
        self.number_of_videos = self._count_raw_videos()
        
        # filter data by downloaded videos
        self.download_video_clean_data = self._filter_downloaded_videos_from_clean_dataset()

        if (self.verbose):
            print(f"Glossary for {self.glossary_clean_data[0]["gloss"]} pre Video Filteration: {len(self.glossary_clean_data[0]["instances"])}, Updated Glossary Size after Video Filteration: {len(self.download_video_clean_data[0]["instances"])}")

        
        if(self.verbose):
            print("\n\n📂 Starting Dataset Generation Process...")

        # generate dataset
        self._generate_dataset()
        print(self.dataset)

        self.save_clean_data_to_json()
        if(self.verbose):
                print("\n\n✅ Dataset Generation Completed Successfully!")

        

        pass

    def _filter_data_by_glossary(self):
        for gloss in self.raw_data:
            if gloss["gloss"].upper() in self.glossary:
                self.clean_data.append(gloss)
            pass
        pass

    def get_clean_data(self):
        return self.clean_data

    def _filter_downloaded_videos_from_clean_dataset(self):
        # return a clean_data list that only contains entries with downloaded videos
        filtered_data = []

        for entry in self.glossary_clean_data:
            gloss = entry['gloss']
            instances = entry['instances']

            filtered_instances = []
            for inst in instances:
                video_id = inst['video_id']
                video_path_mp4 = Path(f"raw_videos/{video_id}.mp4")
                video_path_swf = Path(f"raw_videos/{video_id}.swf")

                if video_path_mp4.exists() or video_path_swf.exists():
                    filtered_instances.append(inst)

            if filtered_instances:
                filtered_entry = {
                    'gloss': gloss,
                    'instances': filtered_instances
                }
                filtered_data.append(filtered_entry)

        return filtered_data

        pass

    def _count_raw_videos(self, raw_videos_folder="raw_videos"):
        """
        Count the number of video files in the raw_videos folder.
        
        Args:
            raw_videos_folder (str): Path to the raw videos folder
            
        Returns:
            int: Number of video files found
        """
        # Define common video file extensions
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
        
        # Get the raw_videos folder path
        raw_videos_path = Path(raw_videos_folder)
        
        # Check if the folder exists
        if not raw_videos_path.exists():
            print(f"Warning: Raw videos folder '{raw_videos_folder}' does not exist.")
            return 0
        
        # Count video files
        video_count = 0
        for file_path in raw_videos_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                video_count += 1
        
        print(f"Found {video_count} video files in '{raw_videos_folder}' folder.")
        return video_count

    def get_updated_glossary(self):
        print([entry['gloss'] for entry in self.download_video_clean_data])
        return [entry['gloss'] for entry in self.download_video_clean_data]

    def _generate_dataset(self):
        # generate dataset in this format {prompt, video_path, gloss}
        for entry in self.download_video_clean_data:
            gloss = entry['gloss']
            instances = entry['instances']

            for inst in instances:
                video_id = inst['video_id']
                video_path_mp4 = Path(f"raw_videos/{video_id}.mp4")
                video_path_swf = Path(f"raw_videos/{video_id}.swf")

                if video_path_mp4.exists():
                    video_path = "raw_videos/" + str(video_id) + ".mp4"
                elif video_path_swf.exists():
                    video_path = "raw_videos/" + str(video_id) + ".swf"
                else:
                    continue  # Skip if no video file exists

                prompt = f"You are an expert sign-language recognition model. Identify the sign in the video and respond with exactly one word and nothing else."

                data_entry = {
                    "prompt": prompt,
                    "video_path": video_path,
                    "gloss": gloss
                }

                self.dataset.append(data_entry)
        
        pass
    
    def save_clean_data_to_json(self, output_file="wlasl_cleaned.json"):
        # Create datasets directory if it doesn't exist
        datasets_dir = Path("datasets")
        datasets_dir.mkdir(exist_ok=True)
        
        with open("datasets/" + output_file, "w") as file:
            json.dump(self.dataset, file, indent=4)

        pass

    
    pass



if __name__ == "__main__":
    data_cleaner = DataCleaner()