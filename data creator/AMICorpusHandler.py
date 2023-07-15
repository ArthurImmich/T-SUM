import os
from pathlib import Path
import urllib.request
import zipfile
from collections import defaultdict
import xml.etree.ElementTree as ET
import re
from tqdm import tqdm
from datasets import Dataset


class AMICorpusHandler:
    def __init__(self, ami_corpus_dir):
        self.ami_dir = os.path.join(ami_corpus_dir, "ami_public_manual_1.6.2")
        self.words_dir = os.path.join(self.ami_dir, "words")
        self.segments_dir = os.path.join(self.ami_dir, "segments")
        self.dialogue_acts_dir = os.path.join(self.ami_dir, "dialogueActs")
        self.extractive_dir = os.path.join(self.ami_dir, "extractive")
        self.abstractive_dir = os.path.join(self.ami_dir, "abstractive")
        self.topic_dir = os.path.join(self.ami_dir, "topics")
        self.__download_corpus()

    def __download_corpus(self):
        download_link = "http://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
        if not os.path.exists(self.ami_dir):
            Path(self.ami_dir).mkdir(parents=True, exist_ok=True)
            print(f"Downloading AMI Corpus to: {self.ami_dir}")
            zipped_ami_filename = f"{self.ami_dir}.zip"
            urllib.request.urlretrieve(download_link, zipped_ami_filename)
            with zipfile.ZipFile(zipped_ami_filename, "r") as zip_ref:
                zip_ref.extractall(self.ami_dir)
            os.remove(zipped_ami_filename)
        else:
            print(f"AMI Corpus has already been downloaded in: {self.ami_dir}")

    def __get_file_and_ids(self, href):
        filename = href.split("#")[0]
        ids = re.findall(r"\((.*?)\)", href)
        if len(ids) > 1:
            start, end = ids
            return filename, start, end
        return filename, ids[0], ids[0]

    def __get_word_index(self, word_id):
        return int(re.findall(r"\d+", word_id.split(".")[-1])[0])

    def __group_meetings_files(self):
        print("Grouping meetings files...")

        transcript_files = [f for f in os.listdir(self.words_dir) if f.endswith(".xml")]

        transcript_files.extend(
            [
                f
                for f in os.listdir(self.dialogue_acts_dir)
                if f.endswith("dialog-act.xml")
            ]
        )

        transcript_files.extend([f for f in os.listdir(self.segments_dir)])

        summary_files = [
            f
            for l in [
                os.listdir(self.extractive_dir),
                os.listdir(self.abstractive_dir),
                os.listdir(self.topic_dir),
            ]
            for f in l
            if f.endswith(".xml")
        ]

        grouped_meeting_files = defaultdict(
            lambda: defaultdict(lambda: defaultdict(defaultdict))
        )

        for f in summary_files:
            meeting, doc_class, _ = f.split(".")
            grouped_meeting_files[meeting][doc_class] = f

        for f in transcript_files:
            meeting, speaker, doc_class, _ = f.split(".")
            if meeting in grouped_meeting_files:
                grouped_meeting_files[meeting]["speakers"][speaker][doc_class] = f

        to_del = []
        for k in grouped_meeting_files.keys():
            if len(grouped_meeting_files[k]) != 5:
                to_del.append(k)

        for k in to_del:
            del grouped_meeting_files[k]

        return grouped_meeting_files

    def __get_meeting_extractive_summary(self, meeting):
        """
        Get a list of binaries where 1 represents that the current sentence is part of the extractive summary for each topic of each meeting.
        :param meeting: dict, the meeting object from meetings[meeting_id]
        :return: list of int, binary values representing extractive summary sentences
        """
        extractive_file = os.path.join(self.extractive_dir, meeting["extsumm"])
        root = ET.parse(extractive_file).getroot()
        for child in root:
            for grandChild in child:
                file, start_id, end_id = self.__get_file_and_ids(
                    grandChild.attrib["href"]
                )
                speaker = file.split(".")[1]
                start_id = int(start_id.split(".")[-1])
                end_id = int(end_id.split(".")[-1]) + 1

                for i in range(start_id, end_id):
                    if i in meeting["speakers"][speaker]["data"]["dialog_acts"].keys():
                        meeting["speakers"][speaker]["data"]["dialog_acts"][i][
                            "extractive"
                        ] = True

        return meeting

    def __get_meeting_speaker_words(self, speaker):
        """
        Get the list of words of a meeting speaker
        :param speaker: dict, dict with speaker data
        :return: list of tuples, words with indexes
        """
        words_file = os.path.join(self.words_dir, speaker["words"])
        root = ET.parse(words_file).getroot()
        length = self.__get_word_index(
            root[-1].attrib["{http://nite.sourceforge.net/}id"]
        )
        speaker["data"]["words"] = [{"i": i, "word": ""} for i in range(length + 1)]

        for w in root:
            i = self.__get_word_index(w.attrib["{http://nite.sourceforge.net/}id"])
            if w.text is not None:
                speaker["data"]["words"][i]["word"] = w.text
                if "starttime" in w.attrib.keys():
                    speaker["data"]["words"][i]["time"] = float(w.attrib["starttime"])

        return speaker

    def __get_meeting_speaker_dialog_act(self, speaker):
        """
        Get the list of acts of each speaker of a meeting
        :param speaker: dict, speaker data
        :return: list of tuples, acts with words indexes
        """

        def __get_dact_time(words):
            for w in words:
                if "time" in w.keys():
                    return w["time"]

        dialog_act_file = os.path.join(self.dialogue_acts_dir, speaker["dialog-act"])
        root = ET.parse(dialog_act_file).getroot()
        speaker["data"]["dialog_acts"] = {}
        for dact in root:
            _, start_id, end_id = self.__get_file_and_ids(
                dact.find("{http://nite.sourceforge.net/}child").attrib["href"]
            )

            start_id = self.__get_word_index(start_id)
            end_id = self.__get_word_index(end_id) + 1

            speaker["data"]["dialog_acts"][
                int(dact.attrib["{http://nite.sourceforge.net/}id"].split(".")[-1])
            ] = {
                "words": speaker["data"]["words"][start_id:end_id],
                "extractive": False,
                "time": __get_dact_time(speaker["data"]["words"][start_id:end_id]),
            }

        return speaker

    def __ordered_speakers_acts_join(self, meeting):
        """
        Get the list of words of each speaker of a meeting
        :param meeting: dict, the meeting object from meetings[meeting_id]
        :return: meeting
        """

        meeting["data"]["dialog_acts"] = [
            dact
            for speaker in meeting["speakers"].values()
            for dact in speaker["data"]["dialog_acts"].values()
            if dact["time"] is not None
        ]

        meeting["data"]["dialog_acts"] = sorted(
            meeting["data"]["dialog_acts"], key=lambda x: x["time"]
        )

        return meeting

    def __get_transcript_from_meeting_acts(self, meeting):
        """
        Transform the meeting acts in a transcript
        :param meeting: dict, the meeting object from meetings[meeting_id]
        :return: meeting
        """
        meeting["data"]["transcript"] = []
        for sentence in meeting["data"]["dialog_acts"]:
            meeting["data"]["transcript"].append(
                {
                    "sentence": " ".join(
                        " ".join([word["word"] for word in sentence["words"]])
                        .strip()
                        .split()
                    ),
                    "extractive": sentence["extractive"]
                    if "extractive" in sentence.keys()
                    else False,
                    "end_topic": any("end_topic" in word for word in sentence["words"]),
                }
            )

        return meeting

    def __get_meeting_abstractive_summary(self, meeting):
        """
        Get the meeting abstractive summary
        :param meeting: dict, the meeting object from meetings[meeting_id]
        :return: meeting
        """
        file = os.path.join(self.abstractive_dir, meeting["abssumm"])
        root = ET.parse(file).getroot()
        sentences = []
        for child in root:
            sentences.extend(child.findall("sentence"))
        meeting["data"]["abstractive"] = {
            sentence.attrib["{http://nite.sourceforge.net/}id"]: {
                "abstract": sentence.text,
                "extract": [],
            }
            for sentence in sentences
        }

        return meeting

    def __link_meeting_extractive_abstractive_summary(self, meeting):
        """
        Link meeting extractive summary with abstractive summary
        :param meeting: dict, the meeting object from meetings[meeting_id]
        :return: meeting
        """
        summlink_file = os.path.join(self.extractive_dir, meeting["summlink"])
        summlink_root = ET.parse(summlink_file).getroot()
        for extractive, abstractive in summlink_root:
            extractive_file, extractive_id, _ = self.__get_file_and_ids(
                extractive.attrib["href"]
            )
            extractive_id = int(extractive_id.split(".")[-1])
            speaker = extractive_file.split(".")[1]

            abstractive_file, abstractive_id, _ = self.__get_file_and_ids(
                abstractive.attrib["href"]
            )

            assert abstractive_file == meeting["abssumm"]

            meeting["data"]["abstractive"][abstractive_id]["extract"].append(
                meeting["speakers"][speaker]["data"]["dialog_acts"][extractive_id]
            )

        for value in meeting["data"]["abstractive"].values():
            value["extract"] = list(
                map(
                    lambda x: {
                        "sentence": " ".join(
                            " ".join([w["word"] for w in x["words"]]).strip().split()
                        ),
                        "time": x["time"],
                    },
                    value["extract"],
                )
            )
            value["extract"] = sorted(value["extract"], key=lambda x: x["time"])

        return meeting

    def __get_meeting_topic_boundaries(self, meeting):
        """
        Get a list of binaries each value representing a single sentence where 1 means the following sentence is a topic boundary.
        Based on the list of sentences/segments extracted in the __get_meeting_transcripted_segments method.
        :param meeting: dict, the meeting object from meetings[meeting_id]
        :return: meeting
        """

        ns = {"nite": "http://nite.sourceforge.net/"}

        def __get_nested_topics(element):
            result = []
            topics = element.findall("topic", ns)
            for topic in topics:
                topic_result = []
                children = topic.findall("nite:child", ns)
                for child in children:
                    topic_result.append(child.get("href"))
                result.append(topic_result)
                result.extend(__get_nested_topics(topic))
            return result

        topic_file = os.path.join(self.topic_dir, meeting["topic"])
        root = ET.parse(topic_file).getroot()
        topics = __get_nested_topics(root)

        for topic in topics:
            if len(topic) > 0:
                file, _, end_id = self.__get_file_and_ids(topic[-1])
                speaker = file.split(".")[1]
                end_id = self.__get_word_index(end_id)
                meeting["speakers"][speaker]["data"]["words"][end_id][
                    "end_topic"
                ] = True

        return meeting

    def __get_meeting_data(self, meeting):
        """
        Get data for a given meeting by calling all the previously defined methods.
        :param meeting: dict, the meeting object from meetings[meeting_id]
        :return: dict, a dictionary with the results of all the methods
        """
        for speaker in meeting["speakers"].keys():
            meeting["speakers"][speaker]["data"] = {}

            meeting["speakers"][speaker] = self.__get_meeting_speaker_words(
                meeting["speakers"][speaker]
            )

            meeting["speakers"][speaker] = self.__get_meeting_speaker_dialog_act(
                meeting["speakers"][speaker]
            )

        meeting = self.__get_meeting_extractive_summary(meeting)

        meeting = self.__get_meeting_topic_boundaries(meeting)

        meeting = self.__ordered_speakers_acts_join(meeting)

        meeting = self.__get_transcript_from_meeting_acts(meeting)

        meeting = self.__get_meeting_abstractive_summary(meeting)

        meeting = self.__link_meeting_extractive_abstractive_summary(meeting)

        return meeting

    def get_all_meetings_data(self):
        """
        Get data for all meetings by calling the get_meeting_data method for each meeting.
        :return: dict, a dictionary with the results of calling get_meeting_data for each meeting
        """
        meetings = self.__group_meetings_files()

        datasets = []
        last_end_topic = None
        for meeting_id in tqdm(meetings.keys()):
            data = self.__get_meeting_data(meetings[meeting_id])
            sentences = []
            topics = []
            extractive = []
            abstractive = []
            for d in data["data"]["transcript"]:
                sentences.append(d["sentence"])
                is_end_topic = 1 if d["end_topic"] else 0
                if last_end_topic == 1 and is_end_topic == 1:
                    if len(topics) > 0:
                        topics[-1] = 0
                last_end_topic = is_end_topic
                topics.append(is_end_topic)
                extractive.append(1 if d["extractive"] else 0)
            for d in data["data"]["abstractive"].values():
                abstract = d["abstract"]
                extract = []
                for s in d["extract"]:
                    extract.append(s["sentence"])
                abstractive.append({"abstract": abstract, "extract": extract})
            datasets.append(
                {
                    "sentences": sentences,
                    "topics": topics,
                    "extractive": extractive,
                    "abstractive": abstractive,
                }
            )

        return Dataset.from_list(datasets).with_format(type="torch")

    def get_data(self):
        """
        Get data for all meetings by calling the get_meeting_data method for each meeting.
        :return: dict, a dictionary with the results of calling get_meeting_data for each meeting
        """
        meetings = self.__group_meetings_files()

        meeting_list = []
        for meeting_id in tqdm(meetings.keys()):
            data = self.__get_meeting_data(meetings[meeting_id])

            meeting = {
                "extract": [],
                "abstract": [],
                "sentences": [],
                "extractive_label": [],
                "end_topic_label": [],
            }

            for topic in data["data"]["abstractive"].values():
                abstract = topic["abstract"]
                extract = sorted(topic["extract"], key=lambda x: x["time"])
                extract = " ".join([sentence["sentence"] for sentence in extract])
                meeting["extract"].append(extract)
                meeting["abstract"].append(abstract)

            for sentence in data["data"]["transcript"]:
                meeting["sentences"].append(sentence["sentence"])
                meeting["extractive_label"].append(1 if sentence["extractive"] else 0)
                meeting["end_topic_label"].append(1 if sentence["end_topic"] else 0)

            meeting_list.append(meeting)

        return Dataset.from_dict({"meetings": meeting_list})
