import json
import os
import sys

# 添加当前目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main(request):
    return {"status": "ok"}
