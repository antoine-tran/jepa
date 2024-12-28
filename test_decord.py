from decord import VideoReader
from decord import cpu, gpu
import sys
video_file = sys.argv[1]

vr = VideoReader(video_file, ctx=cpu(0))
print('video frames:', len(vr))