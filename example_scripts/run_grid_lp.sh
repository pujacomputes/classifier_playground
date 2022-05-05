#lr,aug
sbatch -J "lp30co" -o "z-lp30co.out" run_lp.sh 30 cutout
sbatch -J "lp30mp" -o "z-lp30mp.out" run_lp.sh 30 mixup
sbatch -J "lp30cm" -o "z-lp30cm.out" run_lp.sh 30 cutmix
sbatch -J "lp30aa" -o "z-lp30aa.out" run_lp.sh 30 autoaug 
sbatch -J "lp30am" -o "z-lp30am.out" run_lp.sh 30 augmix 
sbatch -J "lp30ra" -o "z-lp30ra.out" run_lp.sh 30 randaug 