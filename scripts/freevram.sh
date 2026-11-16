#!/usr/bin/env bash
# Run a command with the hazkey IME server (systemd user unit) stopped, restoring it afterwards.
restore=0
if systemctl --user is-active --quiet hazkey-server.service 2>/dev/null; then
  systemctl --user stop hazkey-server.service && restore=1 && sleep 1
fi
trap '[ $restore = 1 ] && systemctl --user start hazkey-server.service' EXIT
"$@"
