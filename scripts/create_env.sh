conda create -y --name waveformer_new python=3.10 pytorch torchvision torchaudio cudatoolkit=11.6 pytorch-lightning tqdm jsonargparse librosa cudatoolkit-dev -c pytorch -c nvidia -c conda-forge &&
conda activate waveformer_new &&
pip3 install pytorch-fast-transformers performer_pytorch audiofile nnAudio pandas neptune-client

