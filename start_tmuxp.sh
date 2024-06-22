#!/bin/bash

# Nome della sessione Tmuxp da verificare (tmux session name)
session_name="bybit"
# Imposta la working directory
cd /opt/directionalscalper

# Verifica se la sessione Tmuxp esiste e la killa se vero
if tmux has-session -t $session_name 2>/dev/null; then
  tmux kill-session -t $session_name
fi

# Avvia tmuxp con il file di configurazione
tmuxp load -d bybit.yaml
