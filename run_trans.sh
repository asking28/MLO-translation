python low_resource_vanilla_bt.py --src_lang=cor --trg_lang=eng --data_name=tatoeba --create_bt_data --tagged --create_path=wiki.aa.cor-eng.eng
python low_resource_vanilla_bt.py --src_lang=deu --trg_lang=ido --data_name=tatoeba --create_bt_data --tagged --create_path=wiki.aa.ido-eng.ido
python low_resource_vanilla_bt.py --src_lang=ido --trg_lang=yid --data_name=tatoeba --create_bt_data --tagged --create_path=wiki.aa.yid-eng.yid
python low_resource_vanilla_bt.py --src_lang=hin --trg_lang=eng --data_name=flores --create_bt_data --tagged --create_path=wiki.aa.cor-eng.eng
python low_resource_vanilla_bt.py --src_lang=cor --trg_lang=eng --data_name=tatoeba --bt --run_all_baselines
python low_resource_vanilla_bt.py --src_lang=deu --trg_lang=ido --data_name=tatoeba --bt --run_all_baselines
python low_resource_vanilla_bt.py --src_lang=ido --trg_lang=yid --data_name=tatoeba --bt --run_all_baselines
python low_resource_vanilla_bt.py --src_lang=hin --trg_lang=eng --data_name=flores --bt --run_all_baselines
python low_resource_vanilla_bt.py --src_lang=cor --trg_lang=eng --data_name=tatoeba --bt
python low_resource_vanilla_bt.py --src_lang=deu --trg_lang=ido --data_name=tatoeba --bt
python low_resource_vanilla_bt.py --src_lang=ido --trg_lang=yid --data_name=tatoeba --bt
python low_resource_vanilla_bt.py --src_lang=hin --trg_lang=eng --data_name=flores --bt
# python low_resource_vanilla_bt.py --src_lang=fr --trg_lang=eng --data_name=multi30 --train_num_points=10000 --create_bt_data --tagged --create_path=wiki.aa.cor-eng.eng
# python low_resource_vanilla_bt.py --src_lang=cs --trg_lang=eng --data_name=multi30 --train_num_points=10000 --create_bt_data --tagged --create_path=wiki.aa.cor-eng.eng
# python low_resource_vanilla_bt.py --src_lang=de --trg_lang=eng --data_name=multi30 --train_num_points=10000 --create_bt_data --tagged --create_path=wiki.aa.cor-eng.eng
# python low_resource_vanilla_bt.py --src_lang=de --trg_lang=eng --data_name=multi30 --train_num_points=10000 --bt --run_all_baselines
# python low_resource_vanilla_bt.py --src_lang=de --trg_lang=eng --data_name=multi30 --train_num_points=10000 --bt 
# python low_resource_vanilla_bt.py --src_lang=fr --trg_lang=eng --data_name=multi30 --train_num_points=10000 --bt --run_all_baselines
# python low_resource_vanilla_bt.py --src_lang=fr --trg_lang=eng --data_name=multi30 --train_num_points=10000 --bt 
# python low_resource_vanilla_bt.py --src_lang=cs --trg_lang=eng --data_name=multi30 --train_num_points=10000 --bt --run_all_baselines
# python low_resource_vanilla_bt.py --src_lang=cs --trg_lang=eng --data_name=multi30 --train_num_points=10000 --bt 
# python betty_low_resource.py --src_lang=fr --trg_lang=eng --train_num_points=30000 --data_name=multi30 --rnn_optimizer=adamw --mbart_optimizer=adamw
# python betty_low_resource.py --src_lang=cor --trg_lang=eng --data_name=tatoeba 
# python betty_low_resource.py --src_lang=deu --trg_lang=ido --data_name=tatoeba 
# python betty_low_resource.py --src_lang=deu --trg_lang=ido --data_name=tatoeba 
# python betty_low_resource.py --src_lang=hin --trg_lang=eng --data_name=flores 
# python betty_low_resource.py --src_lang=bos_Latn --trg_lang=ido --data_name=flores 
# python betty_low_resource.py --src_lang=deu --trg_lang=ido --data_name=flores

# python betty_low_resource.py --src_lang=de --trg_lang=eng --data_name=multi30 --train_num_points=5000 --rnn_optimizer=adamw --clip --darts_adam
# python betty_low_resource.py --src_lang=de --trg_lang=eng --data_name=multi30 --train_num_points=10000 
# python betty_low_resource.py --src_lang=de --trg_lang=eng --data_name=multi30 --train_num_points=30000 
# python betty_low_resource.py --src_lang=fr --trg_lang=eng --data_name=multi30 --train_num_points=5000 
# python betty_low_resource.py --src_lang=fr --trg_lang=eng --data_name=multi30 --train_num_points=10000 
# python betty_low_resource.py --src_lang=fr --trg_lang=eng --data_name=multi30 --train_num_points=30000 
# python betty_low_resource.py --src_lang=cs --trg_lang=eng --data_name=multi30 --train_num_points=5000 
# python betty_low_resource.py --src_lang=cs --trg_lang=eng --data_name=multi30 --train_num_points=10000
# python betty_low_resource.py --src_lang=cs --trg_lang=eng --data_name=multi30 --train_num_points=30000 
# python low_resource_vanilla_bt.py --baseline=swap --baseline_par=6 --src_lang=cor --trg_lang=eng --data_name=tatoba
# python low_resource_vanilla_bt.py --baseline=dropout --baseline_par=0.2 --src_lang=cor --trg_lang=eng --data_name=tatoba
# python low_resource_vanilla_bt.py --baseline=smooth --src_lang=cor --trg_lang=eng --data_name=tatoba
# python low_resource_vanilla_bt.py --baseline=swap --baseline_par=6 --src_lang=deu --trg_lang=ido --data_name=tatoba
# python low_resource_vanilla_bt.py --baseline=dropout --baseline_par=0.2 --src_lang=deu --trg_lang=ido --data_name=tatoba
# python low_resource_vanilla_bt.py --baseline=smooth --src_lang=deu --trg_lang=ido --data_name=tatoba
# python low_resource_vanilla_bt.py --baseline=swap --baseline_par=6 --src_lang=ido --trg_lang=yid --data_name=tatoba
# python low_resource_vanilla_bt.py --baseline=dropout --baseline_par=0.2 --src_lang=ido --trg_lang=yid --data_name=tatoba
# python low_resource_vanilla_bt.py --baseline=smooth --src_lang=ido --trg_lang=yid --data_name=tatoba
# python low_resource_vanilla_bt.py --baseline=swap --baseline_par=6 --src_lang=hin --trg_lang=eng 
# python low_resource_vanilla_bt.py --baseline=dropout --baseline_par=0.2 --src_lang=hin --trg_lang=eng 
# python low_resource_vanilla_bt.py --baseline=smooth --src_lang=hin --trg_lang=eng 
# python low_resource_vanilla_bt.py --baseline=swap --baseline_par=6 --src_lang=bos_Latn --trg_lang=eng 
# python low_resource_vanilla_bt.py --baseline=dropout --baseline_par=0.2 --src_lang=bos_Latn --trg_lang=eng 
# python low_resource_vanilla_bt.py --baseline=smooth --src_lang=bos_Latn --trg_lang=eng 