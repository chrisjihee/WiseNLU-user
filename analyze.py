import itertools
from urllib.request import urlopen

from chrisbase.data import *
from chrisbase.io import *
from chrisbase.util import *
from pandas import DataFrame, Series

logger = logging.getLogger(__name__)
args = CommonArguments(
    env=ProjectEnv(
        project="WiseNLU-user",
        job_name="MorphAnalysis",
        msg_level=logging.INFO,
        msg_format=LoggingFormat.PRINT_00,
    )
)
args.info_args()
# 세종태그셋(2005)
# "NNG", "NNP", "NNB", "NP", "NR", "XR", "XPN", "XSN", "XSV", "XSA",  //10
# "VV", "VA", "VX", "VCP", "VCN", "ETM", "ETN", "EP", "EF", "EC",     //10
# "MM", "JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC",  //10
# "IC", "SH", "SL", "SN", "SS", "SP", "SF", "SE", "SO", "SW",         //10
# "MAG", "MAJ"                                                        //2

tag_groups = {
    "명사": ["NNG", "NNP"],
    "대명사": ["NP"],
    "의존명사": ["NNB"],
    "수사": ["NR"],
    "동사": ["VV"],
    "형용사": ["VA"],
    "관형사": ["MM"],
    "일반부사": ["MAG"],
    "접속부사": ["MAJ"],
    "감탄사": ["IC"],
    "조사": ["JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JX", "JC"],
    "인용격조사": ["JKQ"],
    "어미": ["EP", "EF", "ETN", "ETM"],
    "연결어미": ["EC"],
}


def analyze_text(text, req_id, api_url):
    arg = {
        "request_id": req_id,
        "argument": {
            "text": text,
            "analyzer_types": ["MORPH"],
        }
    }
    f = urlopen(api_url, json.dumps(arg).encode())
    if f.status == 200:
        r = json.loads(f.read().decode())
    if r["request_id"] == req_id:
        doc = r["return_object"]["json"]
    tag_values = {k: [] for k in tag_groups.keys()}
    all_values = []
    for sent in doc["sentence"]:
        ms = [(m["lemma"], m["type"]) for m in sent["morp"]]
        all_values.extend([f"{l}/{t}" for l, t in ms])
        for group, members in tag_groups.items():
            tag_values[group].extend([f"{l}/{t}" for l, t in ms if t in members])
    return tag_values, all_values


def process_file(input_file, output_file, api_url, max_rows=3):
    df: DataFrame = pd.read_excel(input_file, header=0)
    col_names = df.columns.values
    text_columns = df.columns.values[2:]
    outputs = []
    for (_, v) in itertools.islice(df.iterrows(), max_rows):
        v: Series = v
        sid = v[col_names[1]]
        name = v[col_names[0]]
        logger.info(f"- sid={sid}, name={name}")
        for column in text_columns:
            text = v.get(column, "")
            if isinstance(text, float) and math.isnan(text):
                text = ""
            assert isinstance(text, str), f"Invalid type: {type(text)} [text={text}]"
            text = text.strip()
            tag_values, all_values = analyze_text(text, f"{sid}-{column}", api_url)
            output = {
                "번호": sid, "이름": name, "대상": column, "내용": text,
                "전체": ", ".join(all_values),
            }
            for group, values in tag_values.items():
                output[group] = ", ".join(values)
            outputs.append(output)
    pd.DataFrame.from_records(outputs).to_excel(output_file, index=False)


with JobTimer("WritingAnalysis", rt=1, rb=1, rw=80, rc='=', verbose=1):
    process_file(
        input_file="temp/정서 글쓰기와 긍정 부정 글쓰기.xlsx",
        output_file="temp/정서 글쓰기와 긍정 부정 글쓰기 - 분석.xlsx",
        api_url="http://localhost:7100/interface/lm_interface",
        max_rows=300,
    )
