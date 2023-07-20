#!/bin/bash

cd /opt/directionalscalper
# Conta il numero di processi Python in esecuzione (bots)
num_python_processes=$(ps aux | grep "[b]ot.py" -c)

# Numero desiderato dei bot (bots number)
bots_num=12

# Verifica se il numero di processi Python è inferiore a quello desiderato
if [ $num_python_processes -lt $bots_num ]; then
  #echo "Il numero dei bot in esecuzione è inferiore. Eseguo start_tmuxp.sh..."
  # Sostituisci "pippo.sh" con il percorso completo se non si trova nella stessa directory
  ./start_tmuxp.sh
# else
#   echo "Il numero dei bot in esecuzione è corretto."
fi
