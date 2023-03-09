mkdir -p ~/anaconda3 &&
wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh -O ~/anaconda3/ins_conda.sh &&
bash ~/anaconda3/ins_conda.sh -b -u -p ~/anaconda3 &&
~/anaconda3/bin/conda init &&
exec bash && 
./create_env.sh &&
echo "conda activate waveformer_new" >> ~/.bashrc

