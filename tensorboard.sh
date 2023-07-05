#!/usr/bin/bash

#SBATCH --job-name=tensorboard
#SBATCH --partition=any_gpu
#SBATCH --gres=gpu:1
#SBATCH --output /net/talisker/home/benos/mae117/Documents/research/dennis/causal-domain-adaptation/logs/tensorboard/%j.out
#SBATCH --error  /net/talisker/home/benos/mae117/Documents/research/dennis/causal-domain-adaptation/logs/tensorboard/%j.err

# Check if any arguments were supplied
if [ $# -eq 0 ]; then
  # No arguments were supplied, so throw an error
  echo "Error: No arguments were supplied"
  exit 1
fi

# The script was supplied with arguments, so proceed
echo "The following arguments were supplied:"
for arg in "$@"; do
  echo $arg
done

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

tensorboard --logdir $1 --bind_all --port $ipnport
