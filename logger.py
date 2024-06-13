import os
import sys
import logging

log_dir = "logs"
log_filepath = os.path.join(log_dir,"running_logs.log")

os.makedirs(log_dir,exist_ok=True)
 

logging_format = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"


logging.basicConfig(level=logging.INFO,format=logging_format,
                    handlers=[logging.FileHandler(log_filepath),
                              logging.StreamHandler(sys.stdout)],)


logger = logging.getLogger("AI_Evaluation")