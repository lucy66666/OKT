wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1hd3b12cmpMV5e4rHzsjkh1a5hxL-NiqF' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1hd3b12cmpMV5e4rHzsjkh1a5hxL-NiqF" -O results.zip && rm -rf /tmp/cookies.txt
unzip results.zip -d ../results
mv ../results/results/* ../results/
rm -r ../results/results
rm results.zip