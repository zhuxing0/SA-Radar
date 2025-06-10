# train RAD head on RADDet
python train_SA_Radar.py --dataset RADDET --config_dir ./configs/config_simraddet_1000e_final.json --add_train_dir {sim_data_path}(optional)
python validate_SA_Radar.py --config_dir ./configs/config_simraddet_1000e_final.json --dataset RADDET --resume_from {RAD_ckp_path}

# train RAD head on Carrada
python train_SA_Radar.py --dataset CARRADA --config_dir ./configs/config_simcarrada_1000e_final.json --add_train_dir {sim_data_path}(optional)
python validate_SA_Radar.py --config_dir ./configs/config_simcarrada_1000e_final.json --dataset CARRADA --resume_from {RAD_ckp_path}

# train RA head on RADDet (to be release)
python train_cart_SA_Radar.py --dataset RADDET --config_dir ./configs/config_simraddet_1000e_final.json --backbone_resume_from {RAD_ckp_path} --add_train_dir {sim_data_path}(optional)
python validate_cart_SA_Radar.py --dataset RADDET --config_dir ./configs/config_simraddet_1000e_final.json --resume_from {RA_ckp_path}

# train RA head on Carrada (to be release)
python train_cart_SA_Radar.py  --dataset CARRADA --config_dir ./configs/config_simcarrada_1000e_final.json --backbone_resume_from {RAD_ckp_path} --add_train_dir {sim_data_path}(optional)
python validate_cart_SA_Radar.py --dataset CARRADA --config_dir ./configs/config_simcarrada_1000e_final.json --resume_from {RA_ckp_path}

