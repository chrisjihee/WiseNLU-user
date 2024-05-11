from urllib.request import urlopen

from chrisbase.data import *
from chrisbase.io import *
from chrisbase.util import *

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

tag_groups = {
    "명사": ["NNG", "NNP", "NNB"],
    "대명사": ["NP"],
    "수사": ["NR"],
    "동사": ["VV"],
    "형용사": ["VA"],
    "관형사": ["MM"],
    "부사": ["MAG"],
    "감탄사": ["IC"],
    "조사": ["JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JC"],
    "어미": ["EP", "EF", "EC", "ETN", "ETM"],
}

netloc = "localhost:7100"
url = f"http://{netloc}/interface/lm_interface"
with JobTimer("WritingAnalysis", rt=1, rb=1, rw=80, rc='=', verbose=1):
    req_id = "req01"
    arg = {
        "request_id": req_id,
        "argument": {
            "text": "진정한 웃음을 내는 게 아니라고 생각했다. 치열한 다툼이 벌어졌었다. 봄처녀 제 오시네.",
            "analyzer_types": ["MORPH"],
        }
    }
    f = urlopen(url, json.dumps(arg).encode())
    if f.status == 200:
        r = json.loads(f.read().decode())
    if r["request_id"] == req_id:
        doc = r["return_object"]["json"]
    tag_values = {k: [] for k in tag_groups.keys()}
    for sent in doc["sentence"]:
        ms = [(m["lemma"], m["type"]) for m in sent["morp"]]
        for group, members in tag_groups.items():
            tag_values[group].extend([f"{l}/{t}" for l, t in ms if t in members])
    for group, values in tag_values.items():
        print(f"{group}: {', '.join(values)}")
