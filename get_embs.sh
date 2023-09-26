
python gen_embeddings.py --kind=mol --title=cp --cell_profiler
python gen_embeddings.py --kind=mol --title=50_mol --ckpt=embedding_ckpts/50_mol_model.pth --label_prop=0.5
# python gen_embeddings.py --kind=mol --title=best_mol --ckpt=embedding_ckpts/best_mol_model.pth
python gen_embeddings.py --kind=mol --title=best_mol --ckpt=/media/data/sys_ckpts/data_24_model_-40_1.0_1.0_1512_9.pth

