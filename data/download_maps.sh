#! /bin/bash
set -e

git clone https://github.com/jongwook/onsets-and-frames.git
mv onsets-and-frames/data/MAPS .
rm -rf onsets-and-frames

echo
echo MAPS download complete!
