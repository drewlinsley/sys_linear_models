
python gen_embeddings.py --kind=mol --title=cp --cell_profiler
python gen_embeddings.py --kind=mol --title=50_mol --ckpt=50_mol_model.pth --label_prop=0.5
python gen_embeddings.py --kind=mol --title=best_mol --ckpt=best_mol_model.pth

