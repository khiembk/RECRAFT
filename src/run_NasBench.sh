############## NasBench #######################################################################
###1. Darcy 
###1.1 Download dataset.
python DownloadDataset.py --dataset "DARCY-FLOW-5"  
###1.2 Run FLM method.
python newmain.py --config configs/darcy.yaml --seed=0
python newmain.py --config configs/darcy.yaml --seed=1
python newmain.py --config configs/darcy.yaml --seed=2
###1.3 Run ORCA method.
python oldmain.py --config configs/darcy.yaml --seed=0
python oldmain.py --config configs/darcy.yaml --seed=1
python oldmain.py --config configs/darcy.yaml --seed=2
###2.DeepSea.
###2.1 Download dataset.
python DownloadDataset.py --dataset "DEEPSEA"
###2.2 Run FLM method.
python newmain.py --config configs/deepsea.yaml --seed=0
python newmain.py --config configs/deepsea.yaml --seed=1
python newmain.py --config configs/deepsea.yaml --seed=2
###2.3 Run ORCA method.
python oldmain.py --config configs/deepsea.yaml --seed=0
python oldmain.py --config configs/deepsea.yaml --seed=1
python oldmain.py --config configs/deepsea.yaml --seed=2
###3. CIFAR100
###3.1 Download dataset.
python DownloadDataset.py --dataset "CIFAR100"
###3.2 Run FLM method.
python newmain.py --config configs/cifar100.yaml --seed=0
python newmain.py --config configs/cifar100.yaml --seed=1
python newmain.py --config configs/cifar100.yaml --seed=2
###3.3 Run ORCA method.
python oldmain.py --config configs/cifar100.yaml --seed=0
python oldmain.py --config configs/cifar100.yaml --seed=1
python oldmain.py --config configs/cifar100.yaml --seed=2
###4. ECG
###4.1 Download dataset.
python DownloadDataset.py --dataset "ECG"
###4.2 Run FLM method.
python newmain.py --config configs/ecg.yaml --seed=0
python newmain.py --config configs/ecg.yaml --seed=1
python newmain.py --config configs/ecg.yaml --seed=2
###4.3 Run ORCA method.
python oldmain.py --config configs/ecg.yaml --seed=0
python oldmain.py --config configs/ecg.yaml --seed=1
python oldmain.py --config configs/ecg.yaml --seed=2
###5. PSICOV
###5.1 Download dataset.
python DownloadDataset.py --dataset "PSICOV"
###5.2 Run FLM method.
python newmain.py --config configs/psicov.yaml --seed=0
python newmain.py --config configs/psicov.yaml --seed=1
python newmain.py --config configs/psicov.yaml --seed=2
###5.3 Run ORCA method.
python oldmain.py --config configs/psicov.yaml --seed=0
python oldmain.py --config configs/psicov.yaml --seed=1
python oldmain.py --config configs/psicov.yaml --seed=2
###6. NINAPRO
###6.1 Download dataset.
python DownloadDataset.py --dataset "NINAPRO"
###6.2 Run FLM method.
python newmain.py --config configs/ninapro.yaml --seed=0
python newmain.py --config configs/ninapro.yaml --seed=1
python newmain.py --config configs/ninapro.yaml --seed=2
###6.3 Run ORCA method.
python oldmain.py --config configs/ninapro.yaml --seed=0
python oldmain.py --config configs/ninapro.yaml --seed=1
python oldmain.py --config configs/ninapro.yaml --seed=2
###7. COSMIC
###7.1 Download dataset.
python DownloadDataset.py --dataset "COSMIC"
###7.1.5 PREPROCESS DATASET.
python datasets/prepocess_cosmic.py 
###7.2 Run FLM method.
python newmain.py --config configs/cosmic.yaml --seed=0
python newmain.py --config configs/cosmic.yaml --seed=1
python newmain.py --config configs/cosmic.yaml --seed=2
###7.3 Run ORCA method.
python oldmain.py --config configs/cosmic.yaml --seed=0
python oldmain.py --config configs/cosmic.yaml --seed=1
python oldmain.py --config configs/cosmic.yaml --seed=2
###8. FSD
###8.1 Download dataset.
python DownloadDataset.py --dataset "FSD"
###8.2 Run FLM method.
python newmain.py --config configs/fsd.yaml --seed=0
python newmain.py --config configs/fsd.yaml --seed=1
python newmain.py --config configs/fsd.yaml --seed=2
###8.3 Run ORCA method.
python oldmain.py --config configs/fsd.yaml --seed=0
python oldmain.py --config configs/fsd.yaml --seed=1
python oldmain.py --config configs/fsd.yaml --seed=2
###9. SATELLITE
###9.1 Download dataset.
python DownloadDataset.py --dataset "SATELLITE"
###9.2 Run FLM method.
python newmain.py --config configs/satellite.yaml --seed=0
python newmain.py --config configs/satellite.yaml --seed=1
python newmain.py --config configs/satellite.yaml --seed=2
###9.3 Run ORCA method.
python oldmain.py --config configs/satellite.yaml --seed=0
python oldmain.py --config configs/satellite.yaml --seed=1
python oldmain.py --config configs/satellite.yaml --seed=2
###10. SPHERICAL
###10.1 Download dataset.
python DownloadDataset.py --dataset "SPHERICAL"
###10.2 Run FLM method.
python newmain.py --config configs/spherical.yaml --seed=0
python newmain.py --config configs/spherical.yaml --seed=1
python newmain.py --config configs/spherical.yaml --seed=2
###10.3 Run ORCA method.
python oldmain.py --config configs/spherical.yaml --seed=0
python oldmain.py --config configs/spherical.yaml --seed=1
python oldmain.py --config configs/spherical.yaml --seed=2








