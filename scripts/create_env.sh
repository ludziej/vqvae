conda create --name waveformer python=3.10 pytorch torchvision torchaudio cudatoolkit=11.6 pytorch-lightning tqdm jsonargparse librosa -c pytorch -c nvidia -c conda-forge
conda activate waveformer 
pip3 install performer_pytorch audiofile

