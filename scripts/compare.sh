modelpath=/mnt/entropy/vqvae/generated/models/big_vqvae/
infiles=`ls resources/comp | sed  's/^/resources\/comp\//' | tr '\n' ','`
echo $infiles

for model in $@; do
	python3 generate.py --config=gen_big --action=encdec --filepath=\(${infiles:0:-1}\) --vqvae.main_dir=$modelpath/$model --vqvae.restore_ckpt="last_model.ckpt" --out_path=generated/comp/$model/ --level=0
done
