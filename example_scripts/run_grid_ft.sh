#lr,aug
sbatch -J "ft30co" -o "z-ft30co.out" run_ft.sh 0.0003 cutout
sbatch -J "ft30mp" -o "z-ft30mp.out" run_ft.sh 0.0003 mixup
sbatch -J "ft30cm" -o "z-ft30cm.out" run_ft.sh 0.0003 cutmix
sbatch -J "ft30aa" -o "z-ft30aa.out" run_ft.sh 0.0003 autoaug 
sbatch -J "ft30am" -o "z-ft30am.out" run_ft.sh 0.0003 augmix 
sbatch -J "ft30ra" -o "z-ft30ra.out" run_ft.sh 0.0003 randaug 