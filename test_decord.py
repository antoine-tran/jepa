from decord import VideoReader
from decord import cpu, gpu
import sys
video_file = sys.argv[1]

with open(video_file, "rb") as fh:
    vr = VideoReader(fh, ctx=cpu(0))

print('video frames:', len(vr))