#!/bin/bash

ffmpeg -f image2 -framerate 120 -i ../results/demo/demo/frame/%06d.jpg -b 5000k -c:v mpeg4 -r 120 result.mp4
