wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1hqxnYU78WM9kcXui08amcmvEv_9d7IbP' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1hqxnYU78WM9kcXui08amcmvEv_9d7IbP" -O data.zip && rm -rf /tmp/cookies.txt
unzip data.zip -d ../data
mv ../data/data/* ../data/
rm -r ../data/data
rm data.zip