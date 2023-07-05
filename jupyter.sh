#!/usr/bin/bash

#SBATCH --job-name=jupyter
#SBATCH --partition=benos_gpu,dept_gpu
#SBATCH --constraint=A100
#SBATCH --gres=gpu:1
#SBATCH --output /net/talisker/home/benos/mae117/Documents/research/dennis/domain_adaptation/logs/jupyter/%j.out
#SBATCH --error  /net/talisker/home/benos/mae117/Documents/research/dennis/domain_adaptation/logs/jupyter/%j.err

XDG_RUNTIME_DIR=""    

ipnport=$(shuf -i8000-9999 -n1)    
ipnip=$(hostname -i)    
token=$(xxd -l 32 -c 32 -p < /dev/random)    

echo -e "    
Copy/Paste this in your local terminal to ssh tunnel with remote    
-----------------------------------------------------------------    

ssh -N -L $ipnport:$ipnip:$ipnport $USER@cluster
-----------------------------------------------------------------    

Then open a browser on your local machine to the following address    
------------------------------------------------------------------    

http://localhost:$ipnport?token=$token    
------------------------------------------------------------------      
"    

jupyter lab --ZMQChannelsWebsocketConnection.iopub_data_rate_limit=100000000000000 --port=$ipnport --ip=$ipnip --ServerApp.password='' --IdentityProvider.token="$token" --no-browser
