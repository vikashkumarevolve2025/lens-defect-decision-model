"""Download the Roboflow lens_defect dataset (COCO segmentation) into ./data."""
import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("lensdefect").project("lens_defect-fu5gp")
project.version(int(os.getenv("ROBOFLOW_VERSION", 8))).download("coco-segmentation", location="data/lens_defect-8")
