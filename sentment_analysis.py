from pandas import read_excel
import json
from sys import argv
from urllib.request import urlopen

from chrisbase.data import *
from chrisbase.io import *
from chrisbase.util import *

# setup environment
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.ERROR)
args = CommonArguments(
    env=ProjectEnv(
        project="WiseNLU-user",
        job_name="sentment_analysis",
        msg_level=logging.INFO,
        msg_format=LoggingFormat.CHECK_12,
    )
)


def dataframe_to_dict(data, contents_columns):
    result_dict = {}
    for idx, row in data.iterrows():
        row_dict = {col: row[col] for col in contents_columns if pd.notna(row[col])}
        if row_dict:
            result_dict[idx] = row_dict
    return result_dict


def korean_analysis(text, level="WSD", netloc="localhost:7100"):
    url = f"http://{netloc}/interface/lm_interface"
    arg = {
        "request_id": "req01",
        "argument": {
            "text": text,
            "analyzer_types": [level],  # available value: SRL, DPARSE, NER, WSD_POLY, WSD, MORPH
        }
    }
    f = urlopen(url, json.dumps(arg).encode())
    if f.status == 200:
        response = json.loads(f.read().decode())
        document = response["return_object"]["json"]
        sentences = []
        for sentence in document["sentence"]:
            # sent_text = sentence["text"]
            # sent_morps = [f"{x['lemma']}/{x['type']}" for x in sentence["morp"]]
            sent_words = [f"{x['text']}/{x['type']}-{x['scode']}" for x in sentence["WSD"]]
            sentences.append(sent_words)
        return sentences
    else:
        assert False, f"Failed to get response from the server: URL = {url} / status = {f.status} {f.reason}"


if __name__ == '__main__':
    input_file = "data/Emotional-Writing.xlsx"
    output_file = "data/정서 관련 글쓰기 (result).xlsx"

    # read data
    dataframe = pd.read_excel(input_file)
    contents_columns = ["긍정 경험 글", "부정 경험 글", "경험 인식 글", "긍정 경험", "부정 경험"]
    dataframe = dataframe.set_index("번호")
    datadict = dataframe_to_dict(dataframe, contents_columns)

    for idx, row in datadict.items():
        print(f"Processing {idx}...")
        for col in contents_columns:
            if col in row:
                text = row[col]
                result = korean_analysis(text)
