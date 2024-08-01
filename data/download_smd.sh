#! /bin/bash
set -e

echo Downloading the SMD dataset \(3.2 GB\) ...
wget https://zenodo.org/records/10847281/files/SMD-piano_v1.zip

echo Extracting the files ...
mkdir -p SMD
unzip -oq SMD-piano_v1.zip -d SMD
rm SMD-piano_v1.zip

echo 
echo SMD download complete!
