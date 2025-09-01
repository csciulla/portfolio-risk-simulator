import sys
import os

#Set up paths to be reused in each ui file
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
src_path = os.path.join(project_root, 'src')
app_path = os.path.dirname(os.path.dirname(__file__))

if src_path not in sys.path:
    sys.path.insert(0, src_path)
if app_path not in sys.path:
    sys.path.insert(0, app_path)