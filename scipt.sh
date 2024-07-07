
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model LSTM \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --hidden_size 512 \
  --num_layers 2 \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --mark_enc_in 4 \
  --dropout 0.1




python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model xLSTM \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --num_layers 2 \
  --embedding_dim 256 \
  --context_length 96 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --context_length 96 \
  --dropout 0.1



python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model Mamba \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --hidden_size 512 \
  --num_layers 2 \
  --d_model 256 \
  --mark_enc_in 4 \
  --d_state 16 \
  --d_conv 4 \
  --expand 2 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --embedding_dim 256 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --itr 1 \
  --dropout 0.1






python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model S_Mamba \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --d_state 16 \
  --train_epochs 5 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1
