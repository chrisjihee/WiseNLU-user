from chrisbase.io import LoggingFormat
from chrisbase.data import *
from chrisbase.util import *

logger = logging.getLogger(__name__)
args = CommonArguments(
    env=ProjectEnv(
        project="LLM-based",
        job_name="LLaMA-2-13B-Chat",
        msg_level=logging.INFO,
        msg_format=LoggingFormat.PRINT_00,
    )
)

args.info_args()
