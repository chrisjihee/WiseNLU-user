from urllib.request import urlopen
from tqdm import tqdm

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


def dataframe_to_list(data, col):
    result_list = []
    for idx, row in data.iterrows():
        if pd.notna(row[col]):
            result_list.append(row[col])
    return result_list


def dataframe_to_dict(data, label_columns, text_columns):
    result_dict = {}
    for idx, row in data.iterrows():
        row_dict = dict()
        row_dict.update({col: row[col] for col in label_columns})
        row_dict.update({col: row[col] for col in text_columns if pd.notna(row[col])})
        if row_dict:
            result_dict[idx] = row_dict
    return result_dict


def korean_analysis(text, level="MORPH", netloc="localhost:7100"):
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
            if level == "MORPH":
                sent_morps = [f"{x['lemma']}/{x['type']}" for x in sentence["morp"]]
                sentences.append(sent_morps)
            if level == "WSD":
                sent_words = [f"{x['text']}/{x['type']}" for x in sentence["WSD"]]
                # sent_words = [f"{x['text']}/{x['type']}" if x['scode'] == '00' else f"{x['text']}[{x['scode']}]/{x['type']}" for x in sentence["WSD"]]
                sentences.append(sent_words)
        return sentences
    else:
        assert False, f"Failed to get response from the server: URL = {url} / status = {f.status} {f.reason}"


def vocab_raw_to_analized(vocab_raw, level):
    vocab_res = []
    for expr1 in vocab_raw:
        units = []
        if level == "MORPH":
            for s in korean_analysis(expr1, level="MORPH"):
                for w in s:
                    units.append(w)
            expr2 = ' '.join(units)
            expr2 = re.sub(' 다/EF$', '', expr2)
            expr2 = re.sub(' 하/XS[AV]$', '', expr2)
            vocab_res.append(expr2)
        if level == "WSD":
            for s in korean_analysis(expr1, level="WSD"):
                for w in s:
                    units.append(w)
            expr2 = ' '.join(units)
            expr2 = re.sub(r' 다/EF$', '', expr2)
            vocab_res.append(expr2)
    vocab_res = sorted(set(vocab_res))
    return vocab_res


def wsd_to_text(wsd):
    return ''.join(w.split('/')[0] for w in wsd.split())


if __name__ == '__main__':
    vocab_neg_raw_file = "data/Emotional-Vocab-Neg.xlsx"
    vocab_pos_raw_file = "data/Emotional-Vocab-Pos.xlsx"
    vocab_neg_wsd_file = "data/Emotional-Vocab-Neg-anal.xlsx"
    vocab_pos_wsd_file = "data/Emotional-Vocab-Pos-anal.xlsx"

    # read vocab
    if not os.path.exists(vocab_neg_wsd_file):
        vocab_neg = pd.read_excel(vocab_neg_raw_file)
        vocab_neg = vocab_neg.set_index("순번")
        vocab_neg = dataframe_to_list(vocab_neg, "어휘")
        vocab_neg_wsd = vocab_raw_to_analized(vocab_neg, level="WSD")
        vocab_neg_wsd = pd.DataFrame(vocab_neg_wsd, columns=["어휘"])
        vocab_neg_wsd = vocab_neg_wsd.set_index(pd.Index(range(1, len(vocab_neg_wsd) + 1), name="번호"))
        vocab_neg_wsd.to_excel(vocab_neg_wsd_file)
    else:
        vocab_neg_wsd = pd.read_excel(vocab_neg_wsd_file)
        vocab_neg_wsd = vocab_neg_wsd.set_index("번호")
    vocab_neg_wsd = dataframe_to_list(vocab_neg_wsd, "어휘")

    if not os.path.exists(vocab_pos_wsd_file):
        vocab_pos = pd.read_excel(vocab_pos_raw_file)
        vocab_pos = vocab_pos.set_index("순번")
        vocab_pos = dataframe_to_list(vocab_pos, "어휘")
        vocab_pos_wsd = vocab_raw_to_analized(vocab_pos, level="WSD")
        vocab_pos_wsd = pd.DataFrame(vocab_pos_wsd, columns=["어휘"])
        vocab_pos_wsd = vocab_pos_wsd.set_index(pd.Index(range(1, len(vocab_pos_wsd) + 1), name="번호"))
        vocab_pos_wsd.to_excel(vocab_pos_wsd_file)
    else:
        vocab_pos_wsd = pd.read_excel(vocab_pos_wsd_file)
        vocab_pos_wsd = vocab_pos_wsd.set_index("번호")
    vocab_pos_wsd = dataframe_to_list(vocab_pos_wsd, "어휘")

    input_file = "data/Emotional-Writing.xlsx"
    output_file = "data/Emotional-Writing-Analysis.xlsx"

    # read data
    dataframe = pd.read_excel(input_file)
    text_columns = ["긍정 경험 글", "부정 경험 글", "경험 인식 글", "긍정 경험", "부정 경험"]
    label_columns = ["이름"]

    dataframe = dataframe.set_index("번호")
    datadict = dataframe_to_dict(dataframe, label_columns, text_columns)

    # for idx, row in list(datadict.items())[:10]:
    for idx, row in tqdm(datadict.items(), desc="processing student texts"):
        # print(f"Processing {idx}...")
        for col in text_columns:
            if col in row:
                text = row[col]
                text_anal = '\n'.join(' '.join(sent) for sent in korean_analysis(text, level="WSD"))
                neg_counts = {}
                pos_counts = {}
                # print(result)
                for sent in text_anal.split('\n'):
                    for voca in vocab_neg_wsd:
                        neg_counts[voca] = neg_counts.get(voca, 0) + sent.count(voca)
                    for voca in vocab_pos_wsd:
                        pos_counts[voca] = pos_counts.get(voca, 0) + sent.count(voca)
                neg_counts = ', '.join(f"{wsd_to_text(k)}({v})" for k, v in sorted(neg_counts.items(), key=lambda item: item[1], reverse=True) if v != 0)
                pos_counts = ', '.join(f"{wsd_to_text(k)}({v})" for k, v in sorted(pos_counts.items(), key=lambda item: item[1], reverse=True) if v != 0)
                row[f"{col} (anal)"] = text_anal
                row[f"{col} (pos)"] = pos_counts
                row[f"{col} (neg)"] = neg_counts

    # dict to dataframe
    result_df = pd.DataFrame.from_dict(datadict, orient='index')
    result_df = result_df.set_index(pd.Index(range(1, len(result_df) + 1), name="번호"))
    result_df = result_df[
        [
            "이름",
            "긍정 경험 글", "긍정 경험 글 (anal)", "긍정 경험 글 (pos)", "긍정 경험 글 (neg)",
            "부정 경험 글", "부정 경험 글 (anal)", "부정 경험 글 (pos)", "부정 경험 글 (neg)",
            "경험 인식 글", "경험 인식 글 (anal)", "경험 인식 글 (pos)", "경험 인식 글 (neg)",
            "긍정 경험", "긍정 경험 (anal)", "긍정 경험 (pos)", "긍정 경험 (neg)",
            "부정 경험", "부정 경험 (anal)", "부정 경험 (pos)", "부정 경험 (neg)",
        ]
    ]
    result_df.to_excel(output_file)
