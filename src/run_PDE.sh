############## PDEBench #######################################################################
###1. 1DCFD 
###1.1 Download dataset.
python DownloadDataset.py --dataset "PDE1DCFD"  
###1.2 Run FLM method.
python newmain.py --config configs/PDE1DCFD.yaml --seed=0    --pde=True --root_dataset "./PDE_dataset"
python newmain.py --config configs/PDE1DCFD.yaml --seed=1    --pde=True --root_dataset "./PDE_dataset"
python newmain.py --config configs/PDE1DCFD.yaml --seed=2    --pde=True --root_dataset "./PDE_dataset"
###1.3 Run ORCA method.
python oldmain.py --config configs/PDE1DCFD.yaml --seed=0    --pde=True --root_dataset "./PDE_dataset"
python oldmain.py --config configs/PDE1DCFD.yaml --seed=1    --pde=True --root_dataset "./PDE_dataset"
python oldmain.py --config configs/PDE1DCFD.yaml --seed=2    --pde=True --root_dataset "./PDE_dataset"
###2.PDEADV.
###2.1 Download dataset.
python DownloadDataset.py --dataset "PDEADV"
###2.2 Run FLM method.
python newmain.py --config configs/PDEADV.yaml --seed=0      --pde=True --root_dataset "./PDE_dataset"
python newmain.py --config configs/PDEADV.yaml --seed=1      --pde=True --root_dataset "./PDE_dataset"
python newmain.py --config configs/PDEADV.yaml --seed=2      --pde=True --root_dataset "./PDE_dataset"
###2.3 Run ORCA method.
python oldmain.py --config configs/PDEADV.yaml --seed=0      --pde=True --root_dataset "./PDE_dataset"
python oldmain.py --config configs/PDEADV.yaml --seed=1      --pde=True --root_dataset "./PDE_dataset"
python oldmain.py --config configs/PDEADV.yaml --seed=2      --pde=True --root_dataset "./PDE_dataset"
###3. PDEBG
###3.1 Download dataset.
python DownloadDataset.py --dataset "PDEBG"
###3.2 Run FLM method.
python newmain.py --config configs/PDEBG.yaml --seed=0       --pde=True --root_dataset "./PDE_dataset"
python newmain.py --config configs/PDEBG.yaml --seed=1       --pde=True --root_dataset "./PDE_dataset"
python newmain.py --config configs/PDEBG.yaml --seed=2       --pde=True --root_dataset "./PDE_dataset"
###3.3 Run ORCA method.
python oldmain.py --config configs/PDEBG.yaml --seed=0       --pde=True --root_dataset "./PDE_dataset"
python oldmain.py --config configs/PDEBG.yaml --seed=1       --pde=True --root_dataset "./PDE_dataset"
python oldmain.py --config configs/PDEBG.yaml --seed=2       --pde=True --root_dataset "./PDE_dataset"
###4. PDEDS
###4.1 Download dataset.
python DownloadDataset.py --dataset "PDEDS"
###4.2 Run FLM method.
python newmain.py --config configs/PDEDS.yaml --seed=0       --pde=True --root_dataset "./PDE_dataset"
python newmain.py --config configs/PDEDS.yaml --seed=1       --pde=True --root_dataset "./PDE_dataset"
python newmain.py --config configs/PDEDS.yaml --seed=2       --pde=True --root_dataset "./PDE_dataset"
###4.3 Run ORCA method.
python oldmain.py --config configs/PDEDS.yaml --seed=0       --pde=True --root_dataset "./PDE_dataset"
python oldmain.py --config configs/PDEDS.yaml --seed=1       --pde=True --root_dataset "./PDE_dataset"
python oldmain.py --config configs/PDEDS.yaml --seed=2       --pde=True --root_dataset "./PDE_dataset"
###5. PDERD
###5.1 Download dataset.
python DownloadDataset.py --dataset "PDERD"
###5.2 Run FLM method.
python newmain.py --config configs/PDERD.yaml --seed=0       --pde=True --root_dataset "./PDE_dataset"
python newmain.py --config configs/PDERD.yaml --seed=1       --pde=True --root_dataset "./PDE_dataset"
python newmain.py --config configs/PDERD.yaml --seed=2       --pde=True --root_dataset "./PDE_dataset"
###5.3 Run ORCA method.
python oldmain.py --config configs/PDERD.yaml --seed=0       --pde=True --root_dataset "./PDE_dataset"
python oldmain.py --config configs/PDERD.yaml --seed=1       --pde=True --root_dataset "./PDE_dataset"
python oldmain.py --config configs/PDERD.yaml --seed=2       --pde=True --root_dataset "./PDE_dataset"
###6. PDERD2D
###6.1 Download dataset.
python DownloadDataset.py --dataset "PDERD2D"
###6.2 Run FLM method.
python newmain.py --config configs/PDERD2D.yaml --seed=0     --pde=True --root_dataset "./PDE_dataset"
python newmain.py --config configs/PDERD2D.yaml --seed=1     --pde=True --root_dataset "./PDE_dataset"
python newmain.py --config configs/PDERD2D.yaml --seed=2     --pde=True --root_dataset "./PDE_dataset"
###6.3 Run ORCA method.
python oldmain.py --config configs/PDERD2D.yaml --seed=0     --pde=True --root_dataset "./PDE_dataset"
python oldmain.py --config configs/PDERD2D.yaml --seed=1     --pde=True --root_dataset "./PDE_dataset"
python oldmain.py --config configs/PDERD2D.yaml --seed=2     --pde=True --root_dataset "./PDE_dataset"
###7. PDESW
###7.1 Download dataset.
python DownloadDataset.py --dataset "PDESW"
###7.2 Run FLM method.
python newmain.py --config configs/PDESW.yaml --seed=0       --pde=True --root_dataset "./PDE_dataset"
python newmain.py --config configs/PDESW.yaml --seed=1       --pde=True --root_dataset "./PDE_dataset"
python newmain.py --config configs/PDESW.yaml --seed=2       --pde=True --root_dataset "./PDE_dataset"
###7.3 Run ORCA method.
python oldmain.py --config configs/PDESW.yaml --seed=0       --pde=True --root_dataset "./PDE_dataset"
python oldmain.py --config configs/PDESW.yaml --seed=1       --pde=True --root_dataset "./PDE_dataset"
python oldmain.py --config configs/PDESW.yaml --seed=2       --pde=True --root_dataset "./PDE_dataset"
###8. Darcy
###8.1 Download dataset.
python DownloadDataset.py --dataset "PDEDC"
###8.2 Run FLM method.
python newmain.py --config configs/PDEDC.yaml --seed=0       --pde=True --root_dataset "./PDE_dataset"
python newmain.py --config configs/PDEDC.yaml --seed=1       --pde=True --root_dataset "./PDE_dataset"
python newmain.py --config configs/PDEDC.yaml --seed=2       --pde=True --root_dataset "./PDE_dataset"
###8.3 Run ORCA method.
python oldmain.py --config configs/PDEDC.yaml --seed=0       --pde=True --root_dataset "./PDE_dataset"
python oldmain.py --config configs/PDEDC.yaml --seed=1       --pde=True --root_dataset "./PDE_dataset"
python oldmain.py --config configs/PDEDC.yaml --seed=2       --pde=True --root_dataset "./PDE_dataset"






