python betty_low_resource.py --src_lang=cs --trg_lang=eng --train_num_points=30000 --data_name=multi30 --rnn_optimizer=adamw --mbart_optimizer=adamw
# python low_resource_vanilla_bt.py --src_lang=fr --trg_lang=eng --data_name=multi30 --train_num_points=10000 --baseline=smooth
# python low_resource_vanilla_bt.py --src_lang=cs --trg_lang=eng --data_name=multi30 --train_num_points=10000 --baseline=smooth
# python low_resource_vanilla_bt.py --src_lang=de --trg_lang=eng --data_name=multi30 --train_num_points=30000 --baseline=swap --baseline_par=6
# python low_resource_vanilla_bt.py --src_lang=de --trg_lang=eng --data_name=multi30 --train_num_points=30000 --baseline=smooth


