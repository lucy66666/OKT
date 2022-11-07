wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1sKMq5nmOJAGjKP6geAohuekTZbrAzGK2' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1sKMq5nmOJAGjKP6geAohuekTZbrAzGK2" -O pretrained_lm.zip && rm -rf /tmp/cookies.txt
unzip pretrained_lm.zip -d ../model
mv ../model/pretrained_lm/gpt_code_v1 ../model/
mv ../model/pretrained_lm/gpt_code_v1_student ../model/
rm -r ../model/pretrained_lm
rm pretrained_lm.zip