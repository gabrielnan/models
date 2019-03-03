#!/usr/bin/env bash

set -e

FILENAME="challenge_supervised.tar"

echo "Downloading dataset"
 wget --load-cookies /tmp/cookies.txt \
    "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nHLguchtqLlc91NHfpdeuSiB2eesQVb2' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nHLguchtqLlc91NHfpdeuSiB2eesQVb2" \
    -O ${FILENAME} && rm -rf /tmp/cookies.txt

echo "Uncompressing ${FILENAME}"
tar -xf "challenge_supervised.tar"
mv challenge_supervised ctscans
rm ${FILENAME}
