mint \
--host unix:///Users/drj/.orbstack/run/docker.sock \
slim \
--env-file .env \
--http-probe-cmd-file .mint/probeCmds.json \
--include-workdir \
--include-shell \
--include-dir-bins /usr/local/lib/python3.12/site-packages/torch/bin \
--preserve-path /usr/local/lib/python3.12/site-packages/PIL \
serve