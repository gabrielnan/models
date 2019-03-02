#!/usr/bin/env bash


set -e

echo "Downloading dataset"
get --load-cookies /tmp/cookies.txt \
    "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nHLguchtqLlc91NHfpdeuSiB2eesQVb2' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nHLguchtqLlc91NHfpdeuSiB2eesQVb2" \
    -O challenge_supervised.tar && rm -rf /tmp/cookies.txt

tar -xf "ctscans.tar" -O
mv challenge_supervised ctscans
