from pandas import read_excel

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


if __name__ == '__main__':
    input_file = "data/정서 관련 글쓰기_감정분석용.xlsx"
    output_file = "data/정서 관련 글쓰기 (result).xlsx"

    # read data
    dataframe = pd.read_excel(input_file)
    contents_columns = ["긍정 경험 글", "부정 경험 글", "경험 인식 글", "긍정 경험", "부정 경험"]
    dataframe = dataframe.set_index("번호")
    datadict = dataframe_to_dict(dataframe, contents_columns)

